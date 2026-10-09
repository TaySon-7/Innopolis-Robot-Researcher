#!/usr/bin/env python3
"""Run matched autonomous, deterministic-budget and live-LLM Gazebo missions.

Start the stack with LLM=false first. Each model arm runs the production ROS
planner in its own supervised process; it is absent during scenario reset.
Only the llm arm makes external API calls. Results are a descriptive pilot,
not a statistical test of reliability or a comparison with perfect knowledge.
"""

from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import re
import subprocess
import sys
import time
from typing import Any, Callable
from uuid import uuid4

from compare_navigation import API, Comparison, RunError, finite, measurements, metric_delta, positive


ARMS = ('autonomous', 'budget', 'llm')
SCORE_FIELDS = ('scenario', 't', 'battery', 'collected', 'samples_total',
                'distance_travelled', 'collisions', 'false_collects', 'hazard_hits',
                'score', 'finished', 'base_pose', 'world_pose', 'world_pose_valid', 'pose_source')
STATE_FIELDS = ('t', 'episode_id', 'control_mode', 'decision_source', 'navigation',
                'finished', 'pose', 'battery', 'sensor', 'collected', 'samples_total', 'current')

# A supervisor owns the child process group. The stop command verifies both a
# unique token and Linux process start time, avoiding broad pkill/PID reuse.
SUPERVISOR = r'''
import json, os, signal, subprocess, sys, time
token, policy, calls = sys.argv[1:]
root = '/tmp/' + token
deadline = None
child = None
def stop(signum, frame):
    global deadline
    if deadline is None:
        deadline = time.monotonic() + 8
    if child is not None and child.poll() is None:
        try: os.killpg(child.pid, signal.SIGINT)
        except ProcessLookupError: pass
signal.signal(signal.SIGINT, stop)
signal.signal(signal.SIGTERM, stop)
log = open(root + '.ros.log', 'x')
child = subprocess.Popen(['/did-entrypoint.sh', 'ros2', 'run', 'did_llm', 'llm_planner',
    '--ros-args', '-p', 'use_sim_time:=true', '-p', 'selection_policy:=' + policy,
    '-p', 'max_calls_total:=' + calls, '-p', 'exchanges_path:=' + root + '.exchanges.jsonl',
    '-p', 'metrics_path:=' + root + '.metrics.json'],
    stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
start_ticks = open('/proc/%d/stat' % os.getpid()).read().split()[21]
with open(root + '.pid.json', 'x') as handle:
    json.dump({'pid': os.getpid(), 'start_ticks': start_ticks}, handle)
try:
    while child.poll() is None:
        if deadline is not None and time.monotonic() >= deadline:
            os.killpg(child.pid, signal.SIGKILL)
            break
        time.sleep(0.1)
    result = child.wait()
finally:
    if child.poll() is None:
        os.killpg(child.pid, signal.SIGKILL)
        child.wait()
    log.close()
sys.exit(result if result >= 0 else 128 - result)
'''

STOP_SUPERVISOR = r'''
import json, os, signal, sys
token = sys.argv[1]
try:
    owner = json.load(open('/tmp/' + token + '.pid.json'))
    pid = owner['pid']
    stat = open('/proc/%d/stat' % pid).read().split()[21]
    argv = open('/proc/%d/cmdline' % pid, 'rb').read().split(b'\0')
except FileNotFoundError:
    sys.exit(0)
if stat != owner['start_ticks'] or token.encode() not in argv:
    raise SystemExit('owned planner process identity does not match')
os.kill(pid, signal.SIGINT)
'''

READ_METRICS = r'''
import json, math, os, sys
root = '/tmp/' + sys.argv[1]
result = {'ready': False, 'successful_exchanges': 0, 'response_latency_s': [],
          'decision_diagnostics': []}
try:
    data = json.load(open(root + '.metrics.json'))
    result.update({key: data.get(key) for key in ('selection_policy', 'model')})
    result['client_stats'] = {key: data.get('client_stats', {}).get(key) for key in
        ('calls_made', 'cache_hits', 'blocked_by_budget', 'failed')}
    result['ready'] = True
except (FileNotFoundError, ValueError):
    pass
try:
    for line in open(root + '.exchanges.jsonl'):
        try: item = json.loads(line)
        except ValueError: continue
        if not isinstance(item, dict): continue
        value = item.get('seconds')
        if isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value) and value >= 0:
            result['response_latency_s'].append(value)
            result['successful_exchanges'] += 1
        try:
            offer = json.loads(item['user'])['offer']
            feasible = [c for c in offer['candidates'] if c['feasible'] is True]
            searches = [c for c in feasible if c['kind'] == 'search']
            # Exactly experimental_selector.budget_choice's ordering. Kept
            # local so this metrics reader needs no ROS/Python overlay.
            budget = min(searches or feasible, key=lambda c: (c['required_battery'], c['goal_id']))
            chosen = next(c for c in feasible if c['goal_id'] == item['parsed']['goal_id'])
            if not isinstance(offer['snapshot_id'], str) or not isinstance(chosen['goal_id'], str):
                continue
            evidence = chosen.get('evidence')
            result['decision_diagnostics'].append({
                'snapshot_id': offer['snapshot_id'], 'goal_id': chosen['goal_id'],
                'feasible_search_count': len(searches), 'budget_choice_id': budget['goal_id'],
                'llm_differs_from_budget': chosen['goal_id'] != budget['goal_id'],
                'selected_evidence': evidence if evidence in ('observed_signal', 'exploration', 'return') else None,
            })
        except (ValueError, TypeError, KeyError, StopIteration):
            pass
except FileNotFoundError:
    pass
print(json.dumps(result))
'''


class DockerPlanner:
    """Own one planner only. Never starts/stops the shared Compose stack."""

    def __init__(self, token: str, policy: str, max_calls: int = 60) -> None:
        self.token, self.policy, self.max_calls = token, policy, max_calls
        self.process: subprocess.Popen | None = None

    @staticmethod
    def command(script: str, *args: str) -> list[str]:
        return ['docker', 'compose', 'exec', '-T', 'sim', 'python3', '-u', '-c', script, *args]

    def metrics(self) -> dict:
        try:
            response = subprocess.run(self.command(READ_METRICS, self.token), check=True,
                                      capture_output=True, text=True, timeout=15)
            return json.loads(response.stdout)
        except (OSError, subprocess.SubprocessError, ValueError) as error:
            raise RunError('could not read sanitized planner metrics') from error

    def start(self) -> None:
        self.process = subprocess.Popen(
            self.command(SUPERVISOR, self.token, self.policy, str(self.max_calls)),
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        deadline = time.monotonic() + 30
        while time.monotonic() < deadline:
            if self.process.poll() is not None:
                raise RunError(f'{self.policy} planner exited during startup; inspect /tmp/{self.token}.ros.log in sim')
            metrics = self.metrics()
            if metrics.get('ready') and metrics.get('selection_policy') == self.policy:
                return
            time.sleep(0.5)
        raise RunError('planner did not publish startup metrics within 30 wall seconds')

    def alive(self) -> bool:
        return self.process is not None and self.process.poll() is None

    def stop(self) -> dict:
        if self.process is not None and self.process.poll() is None:
            try:
                subprocess.run(self.command(STOP_SUPERVISOR, self.token), check=True,
                               capture_output=True, text=True, timeout=15)
                self.process.wait(timeout=15)
            except (OSError, subprocess.SubprocessError) as error:
                raise RunError('could not confirm owned planner process stopped; abort comparison') from error
        return self.metrics()


def schedule(scenarios: list[str], arms: tuple[str, ...] = ARMS) -> list[tuple[str, str]]:
    """Rotate order by scenario to avoid putting the LLM last on every level."""
    return [(scenario, arm) for index, scenario in enumerate(scenarios)
            for arm in arms[index % len(arms):] + arms[:index % len(arms)]]


def clean_score(snapshot: dict) -> dict:
    score = snapshot.get('score') or {}
    return {key: score[key] for key in SCORE_FIELDS if key in score}


def summary(start: dict, end: dict) -> dict:
    score = clean_score(end)
    delta = metric_delta(measurements(start), measurements(end))
    base, pose = score.get('base_pose', {}), score.get('world_pose', {})
    distance = (math.hypot(pose['x'] - base['x'], pose['y'] - base['y'])
                if all(finite(point.get(axis)) for point in (base, pose) for axis in ('x', 'y')) else None)
    return {**delta, **{key: score.get(key) for key in
            ('collected', 'samples_total', 'score', 'false_collects', 'hazard_hits', 'battery', 'finished')},
            'returned_to_base': score.get('finished') is True,
            'distance_from_base_m': None if distance is None else round(distance, 4),
            'episode_simulation_s': score['t']}


class MissionComparison(Comparison):
    def __init__(self, api: API, *, sim_limit: float = 900, wall_limit: float = 1800,
                 poll: float = 0.5, ready_timeout: float = 60,
                 planner_factory: Callable = DockerPlanner, clock: Callable = time.monotonic,
                 sleep: Callable = time.sleep, progress: Callable = print) -> None:
        super().__init__(api, poll=poll, ready_timeout=ready_timeout, goal_timeout=sim_limit,
                         wall_timeout=wall_limit, tolerance=0.2, clock=clock, sleep=sleep)
        self.planner_factory, self.progress = planner_factory, progress

    def run_mission(self, arm: str, scenario: str, token: str, raw_path: Path,
                    max_calls: int = 60) -> dict:
        result: dict = {'arm': arm, 'scenario': scenario, 'backend': 'custom',
                        'valid': False, 'censored': False, 'raw_path': str(raw_path.resolve())}
        planner = None
        start = end = None
        seen_journal: set[str] = set()
        decisions: dict[str, str] = {}
        accepted: set[str] = set()
        rejected: set[str] = set()
        terminals: dict[str, str] = {}
        previous: dict[str, Any] = {}
        wall_start = self.clock()

        def record(snapshot: dict, handle) -> None:
            nonlocal end
            end = snapshot
            state = snapshot.get('state') or {}
            handle.write(json.dumps({'kind': 'telemetry', 'wall_elapsed_s': self.clock() - wall_start,
                                    'score': clean_score(snapshot),
                                    'state': {key: state[key] for key in STATE_FIELDS if key in state}},
                                   ensure_ascii=False, allow_nan=False) + '\n')
            for entry in snapshot.get('journal') or []:
                key = json.dumps(entry, sort_keys=True, ensure_ascii=False)
                if key not in seen_journal:
                    seen_journal.add(key)
                    handle.write(json.dumps({'kind': 'journal', 'entry': entry}, ensure_ascii=False) + '\n')
                    if entry.get('source') == 'llm_math_planner' and entry.get('decision_source'):
                        decisions[entry['plan_id']] = entry['decision_source']
                    if (entry.get('source') == 'llm_math_planner'
                            and entry.get('execution_state') in ('done', 'failed', 'preempted')):
                        accepted.add(entry['plan_id'])
                        terminals[entry['plan_id']] = entry['execution_state']
            status = snapshot.get('status') or {}
            identifier = status.get('plan_id', '')
            if identifier.startswith('math-'):
                if (status.get('data') or {}).get('accepted') is False:
                    rejected.add(identifier)
                elif status.get('state') in ('running', 'done', 'failed', 'preempted'):
                    accepted.add(identifier)
                    if status.get('plan_complete') is True:
                        terminals[identifier] = status['state']
            for key in ('plan', 'status'):
                value = snapshot.get(key)
                if value != previous.get(key):
                    handle.write(json.dumps({'kind': key, 'value': value}, ensure_ascii=False) + '\n')
                    previous[key] = value
            handle.flush()

        with raw_path.open('x', encoding='utf-8') as raw:
            try:
                initial = self.prepare('custom', scenario)
                if any(initial['score'].get(key) != 0 for key in ('collected', 'false_collects', 'hazard_hits')):
                    raise RunError('scenario reset did not clear mission counters')
                result['episode_id'] = initial['state']['episode_id']
                result['reset_ready_simulation_s'] = initial['score']['t']
                if arm != 'autonomous':
                    planner = self.planner_factory(token, arm, max_calls)
                    planner.start()
                start = self.snapshot()
                if start['state']['episode_id'] != result['episode_id']:
                    raise RunError('episode changed during planner startup')
                if start['state'].get('control_mode') != 'stopped':
                    raise RunError('another controller started during preparation')
                if not finite(start['state'].get('t')):
                    raise RunError('agent state timestamp is unavailable')
                result['start_score'] = clean_score(start)
                wall_start = self.clock()
                self.api.request('command', {'cmd': 'auto' if arm == 'autonomous' else 'llm'})
                last_progress = wall_start - 30
                last_telemetry_t, last_telemetry_wall = start['score']['t'], wall_start
                last_agent_t, last_agent_wall = start['state']['t'], wall_start
                terminal_at = None
                command_acknowledged = False
                while True:
                    snapshot = self.snapshot()
                    state, score = snapshot.get('state', {}), snapshot.get('score', {})
                    if state.get('episode_id') != result['episode_id'] or score.get('scenario') != scenario:
                        raise RunError('episode/scenario changed during mission')
                    if not self.backend_ready(snapshot, 'custom'):
                        raise RunError('navigation backend changed or became unavailable')
                    agent_t = state.get('t')
                    if not finite(agent_t):
                        raise RunError('agent state timestamp is unavailable')
                    if agent_t < last_agent_t:
                        raise RunError('agent state clock moved backwards during mission')
                    if agent_t > last_agent_t:
                        last_agent_t, last_agent_wall = agent_t, self.clock()
                    elif self.clock() - last_agent_wall >= 30:
                        raise RunError('agent state telemetry did not advance for 30 wall seconds')
                    allowed = ('autonomous',) if arm == 'autonomous' else ('llm', 'fallback')
                    # State is published at 1 Hz in simulation time. A slow
                    # Gazebo run can take several wall seconds for one update.
                    # Grace applies only before the first acknowledgement.
                    if state.get('control_mode') in allowed:
                        command_acknowledged = True
                    elif (command_acknowledged or self.clock() - wall_start >= 30
                          or state.get('control_mode') != 'stopped'):
                        raise RunError('another controller took over or command was not acknowledged')
                    measurements(snapshot)
                    if not all(finite(score.get(key)) for key in
                               ('collected', 'samples_total', 'score', 'false_collects', 'hazard_hits')):
                        raise RunError('required mission score counters are unavailable')
                    if score['t'] > last_telemetry_t:
                        last_telemetry_t, last_telemetry_wall = score['t'], self.clock()
                    elif self.clock() - last_telemetry_wall >= 30:
                        raise RunError('judge simulation telemetry did not advance for 30 wall seconds')
                    metric_delta(measurements(start), measurements(snapshot))
                    if end is not None:
                        metric_delta(measurements(end), measurements(snapshot))
                    record(snapshot, raw)
                    if self.clock() - last_progress >= 30:
                        self.progress(f'{scenario} / {arm}: t={score["t"]:.1f}s '
                                      f'samples={score.get("collected")}/{score.get("samples_total")} '
                                      f'battery={score["battery"]:.2f} score={score.get("score")}')
                        last_progress = self.clock()
                    if score.get('finished') is True:
                        result.update(valid=True, outcome='finished', finish_reason='judge finished at base')
                        break
                    if score['battery'] <= 0:
                        result.update(valid=True, outcome='battery_depleted', finish_reason='battery depleted')
                        break
                    if score['t'] >= self.goal_timeout:
                        result.update(valid=True, censored=True, outcome='simulation_timeout',
                                      finish_reason='episode simulation horizon reached')
                        break
                    if self.clock() - wall_start >= self.wall_timeout:
                        result.update(valid=True, censored=True, outcome='wall_timeout',
                                      finish_reason='mission wall-clock horizon reached')
                        break
                    if planner is not None and not planner.alive():
                        raise RunError('owned planner process exited during mission')
                    status = snapshot.get('status') or {}
                    if (arm == 'autonomous' and status.get('plan_id') == 'auto'
                            and status.get('plan_complete') is True
                            and status.get('state') in ('done', 'failed', 'preempted')):
                        if terminal_at is None:
                            terminal_at = score['t']
                        if score['t'] > terminal_at:
                            result.update(valid=True, outcome='autonomous_terminal',
                                          finish_reason=status.get('reason', status['state']))
                            break
                    self.sleep(self.poll)
            except (RunError, OSError, ValueError) as error:
                result.update(outcome='invalid', error=str(error), finish_reason=str(error))
            except KeyboardInterrupt:
                result.update(outcome='interrupted', interrupted=True, finish_reason='interrupted by operator')
            finally:
                result['wall_elapsed_s'] = round(self.clock() - wall_start, 3)
                if start is not None and end is not None:
                    result['end_score'] = clean_score(end)
                    try:
                        result['metrics'] = summary(start, end)
                    except RunError as error:
                        result.update(valid=False, error=str(error))
                result['decisions_published'] = dict(Counter(decisions.values()))
                result['decisions_accepted_observed'] = dict(Counter(
                    decisions[identifier] for identifier in accepted - rejected if identifier in decisions))
                result['decisions_rejected_observed'] = dict(Counter(
                    decisions[identifier] for identifier in rejected if identifier in decisions))
                result['terminal_plans_observed'] = {
                    source: dict(Counter(terminals[identifier] for identifier in terminals
                                         if decisions.get(identifier) == source and identifier not in rejected))
                    for source in ('llm', 'budget', 'fallback')}
                result['decision_count_note'] = 'Published from planner journal; accepted/terminal counts are observed lower bounds at polling frequency.'
                count = len(decisions)
                result['llm_decision_share'] = sum(value == 'llm' for value in decisions.values()) / count if count else None
                result['fallback_decision_share'] = sum(value == 'fallback' for value in decisions.values()) / count if count else None
                try:
                    self.stop()
                    result['stop_confirmed'] = True
                except RunError as error:
                    result.update(stop_confirmed=False, valid=False, stop_error=str(error))
                if planner is not None:
                    try:
                        result['planner_metrics'] = planner.stop()
                        result['planner_stop_confirmed'] = True
                    except RunError as error:
                        result.update(planner_stop_confirmed=False, valid=False, planner_stop_error=str(error))
                raw.write(json.dumps({'kind': 'result', 'result': result}, ensure_ascii=False, allow_nan=False) + '\n')
        return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--api', default='http://127.0.0.1:8082')
    parser.add_argument('--scenarios', nargs='+', default=['easy@7', 'medium@7', 'hard@7'])
    parser.add_argument('--arms', nargs='+', choices=ARMS, default=list(ARMS))
    parser.add_argument('--output-dir', type=Path, required=True, help='new directory; existing results are preserved')
    parser.add_argument('--llm-disabled', action='store_true', required=True,
                        help='assert the shared stack was launched with LLM=false')
    parser.add_argument('--sim-limit', type=positive, default=900, help='absolute judge episode seconds')
    parser.add_argument('--wall-limit', type=positive, default=1800, help='wall seconds from command')
    parser.add_argument('--poll', type=positive, default=0.5)
    parser.add_argument('--ready-timeout', type=positive, default=60)
    parser.add_argument('--max-calls', type=int, default=60, help='maximum actual API attempts per llm mission')
    args = parser.parse_args(argv)
    if (any(not re.fullmatch(r'(easy|medium|hard)(@\d+)?', value) for value in args.scenarios)
            or len(set(args.scenarios)) != len(args.scenarios)):
        parser.error('provide distinct easy/medium/hard scenarios, optionally @nonnegative_seed')
    if len(set(args.arms)) != len(args.arms) or args.max_calls <= 0:
        parser.error('arms must be distinct and max-calls must be positive')
    try:
        args.output_dir.mkdir(parents=True, exist_ok=False)
    except OSError as error:
        parser.error(str(error))
    result: dict = {'schema': 'planner-comparison@1', 'run_id': uuid4().hex[:12],
                    'created_at': datetime.now(timezone.utc).isoformat(), 'backend': 'custom',
                    'llm_disabled_asserted': True,
                    'limits': {'episode_simulation_s': args.sim_limit, 'mission_wall_s': args.wall_limit,
                               'api_attempts_per_llm_run': args.max_calls, 'poll_wall_s': args.poll},
                    'schedule': schedule(args.scenarios, tuple(args.arms)), 'runs': []}
    runner = MissionComparison(API(args.api), sim_limit=args.sim_limit, wall_limit=args.wall_limit,
                               poll=args.poll, ready_timeout=args.ready_timeout,
                               progress=lambda text: print(text, flush=True))
    output = args.output_dir / 'results.json'

    def save() -> None:
        temp = output.with_suffix('.tmp')
        temp.write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + '\n', encoding='utf-8')
        temp.replace(output)

    save()
    for index, (scenario, arm) in enumerate(result['schedule']):
        token = f'planner-comparison-{result["run_id"]}-{index}'
        print(f'Start {index + 1}/{len(result["schedule"])}: {scenario} / {arm}', flush=True)
        run = runner.run_mission(arm, scenario, token, args.output_dir / f'{index + 1:02d}-{scenario}-{arm}.jsonl', args.max_calls)
        result['runs'].append(run)
        save()
        print(f'{scenario} / {arm}: {run.get("outcome")} {run.get("metrics", {})}', flush=True)
        if (not run.get('stop_confirmed') or run.get('planner_stop_confirmed') is False
                or run.get('interrupted') or not run.get('valid')):
            break
    result['complete'] = len(result['runs']) == len(result['schedule']) and all(run['valid'] for run in result['runs'])
    save()
    print(str(output.resolve()), flush=True)
    return 0 if result['complete'] else 1


if __name__ == '__main__':
    sys.exit(main())
