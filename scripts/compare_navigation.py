#!/usr/bin/env python3
"""Run identical manual waypoint routes against the live navigation backends.

Requires an already running Gazebo stack launched with LLM=false. Uses only the
public HTTP API and Python's standard library; never starts containers or LLMs.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import re
import sys
import time
from typing import Any, Callable
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen
from uuid import uuid4


class RunError(RuntimeError):
    pass


def finite(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def route_from_json(text: str) -> list[list[float]]:
    route = json.loads(text)
    if not isinstance(route, list) or not 1 <= len(route) <= 50:
        raise ValueError('waypoints must be a JSON list of 1–50 [x, y] points')
    if any(not isinstance(p, list) or len(p) != 2 or not all(finite(v) for v in p)
           for p in route):
        raise ValueError('each waypoint must contain two finite numbers')
    return [[float(x), float(y)] for x, y in route]


class API:
    def __init__(self, base_url: str, timeout: float = 15.0) -> None:
        self.base_url = base_url.rstrip('/')
        self.timeout = timeout

    def request(self, path: str, body: dict | None = None) -> dict:
        data = None if body is None else json.dumps(body, allow_nan=False).encode()
        request = Request(self.base_url + '/api/' + path, data=data,
                          headers={'Content-Type': 'application/json'})
        try:
            with urlopen(request, timeout=self.timeout) as response:
                result = json.load(response)
        except (HTTPError, URLError, TimeoutError, ValueError) as error:
            raise RunError(f'{path}: {error}') from error
        if not isinstance(result, dict) or result.get('ok') is False:
            raise RunError(f'{path}: {result.get("error", "invalid response") if isinstance(result, dict) else "invalid response"}')
        return result


def measurements(snapshot: dict) -> dict:
    """Use judge telemetry for evaluation; never feed hidden samples to plans."""
    score = snapshot.get('score', {})
    values = {key: score.get(key) for key in
              ('t', 'distance_travelled', 'battery', 'collisions')}
    pose = score.get('world_pose') or {}
    if (score.get('world_pose_valid') is not True
            or not all(finite(value) for value in values.values())
            or not all(finite(pose.get(key)) for key in ('x', 'y'))):
        raise RunError('fresh judge world pose and score metrics are required')
    return {**values, 'pose': {'x': pose['x'], 'y': pose['y']}}


def metric_delta(start: dict, end: dict) -> dict:
    result = {
        'simulation_elapsed_s': end['t'] - start['t'],
        'distance_m': end['distance_travelled'] - start['distance_travelled'],
        'battery_used': start['battery'] - end['battery'],
        'collisions': end['collisions'] - start['collisions'],
    }
    if any(value < -0.01 for value in result.values()):
        raise RunError('episode metrics moved backwards; comparison was interrupted')
    return {key: round(value, 4) for key, value in result.items()}


class Comparison:
    def __init__(self, api: API, *, poll: float, ready_timeout: float,
                 goal_timeout: float, wall_timeout: float, tolerance: float,
                 clock: Callable = time.monotonic, sleep: Callable = time.sleep) -> None:
        self.api, self.poll = api, poll
        self.ready_timeout, self.goal_timeout = ready_timeout, goal_timeout
        self.wall_timeout, self.tolerance = wall_timeout, tolerance
        self.clock, self.sleep = clock, sleep
        self.latest: dict = {}

    def snapshot(self) -> dict:
        self.latest = self.api.request('state')
        return self.latest

    def wait(self, predicate: Callable[[dict], bool], reason: str) -> dict:
        deadline = self.clock() + self.ready_timeout
        while self.clock() < deadline:
            snapshot = self.snapshot()
            if predicate(snapshot):
                return snapshot
            self.sleep(self.poll)
        raise RunError(f'timed out waiting for {reason}')

    def stop(self) -> dict:
        self.api.request('command', {'cmd': 'stop'})
        return self.wait(lambda s: s.get('state', {}).get('control_mode') == 'stopped'
                         and s.get('status', {}).get('state') != 'running', 'confirmed stop')

    @staticmethod
    def backend_ready(snapshot: dict, backend: str) -> bool:
        nav = snapshot.get('state', {}).get('navigation') or {}
        return nav.get('backend') == backend and nav.get('ready') is True

    def prepare(self, backend: str, scenario: str) -> dict:
        before = self.stop()
        episode = before.get('state', {}).get('episode_id')
        if not isinstance(episode, int):
            raise RunError('agent episode_id is unavailable')
        self.api.request('navigation', {'backend': backend})
        self.wait(lambda s: self.backend_ready(s, backend), f'{backend} readiness')
        self.api.request('scenario', {'scenario': scenario})
        # Reset may replace a previously issued stop. Fence it again after the
        # new agent episode is visible. LLM must be disabled at process launch.
        self.api.request('command', {'cmd': 'stop'})
        self.wait(lambda s: isinstance(s.get('state', {}).get('episode_id'), int)
                  and s['state']['episode_id'] != episode
                  and s.get('score', {}).get('scenario') == scenario,
                  'new scenario episode')
        self.stop()

        def clean(snapshot: dict) -> bool:
            if not self.backend_ready(snapshot, backend):
                return False
            try:
                metrics = measurements(snapshot)
            except RunError:
                return False
            base = snapshot['score'].get('base_pose') or {}
            return (all(finite(base.get(key)) for key in ('x', 'y'))
                    and math.hypot(metrics['pose']['x'] - base['x'],
                                   metrics['pose']['y'] - base['y']) <= 0.05
                    and metrics['distance_travelled'] <= 0.05
                    and metrics['collisions'] == 0
                    and snapshot['score'].get('finished') is False
                    and snapshot['state'].get('control_mode') == 'stopped')

        fresh = self.wait(clean, 'clean reset at base with navigation ready')
        # Require an advancing score timestamp, not a cached reset snapshot.
        return self.wait(lambda s: clean(s) and s['score']['t'] > fresh['score']['t'],
                         'advancing simulation telemetry')

    def goal(self, backend: str, episode: int, target: list[float], plan_id: str) -> dict:
        start = measurements(self.snapshot())
        wall_start = self.clock()
        self.api.request('plan', {'plan_id': plan_id, 'source': 'manual',
                                 'subgoals': [{'type': 'goto', 'x': target[0], 'y': target[1]}]})
        terminal_at = None
        while self.clock() - wall_start < self.wall_timeout:
            snapshot = self.snapshot()
            state = snapshot.get('state', {})
            if state.get('episode_id') != episode:
                raise RunError('episode changed during a waypoint')
            if not self.backend_ready(snapshot, backend):
                raise RunError('navigation backend changed or became unavailable')
            if state.get('control_mode') not in ('manual', 'stopped'):
                raise RunError('another controller took over the robot')
            end = measurements(snapshot)
            elapsed = end['t'] - start['t']
            if elapsed < 0:
                raise RunError('simulation episode clock moved backwards')
            status = snapshot.get('status') or {}
            # Exact IDs prevent a previous run's terminal status being counted.
            if status.get('plan_id') == plan_id and status.get('state') in ('done', 'failed', 'preempted'):
                if status.get('state') != 'done':
                    raise RunError(f'{status["state"]}: {status.get("reason", "unspecified failure")}')
                if status.get('plan_complete') is True:
                    if terminal_at is None:
                        terminal_at = end['t']
                    # Judge metrics and agent status arrive independently.
                    if end['t'] > terminal_at:
                        error = math.hypot(end['pose']['x'] - target[0], end['pose']['y'] - target[1])
                        if error > self.tolerance:
                            raise RunError(f'goal error {error:.3f} m exceeds tolerance {self.tolerance:.3f} m')
                        return {'plan_id': plan_id, 'target': target, 'success': True,
                                'goal_error_m': round(error, 4), 'end_pose': end['pose'],
                                'wall_elapsed_s': round(self.clock() - wall_start, 3),
                                **metric_delta(start, end)}
            if snapshot.get('score', {}).get('finished') is True:
                raise RunError('judge episode finished before route completion')
            if elapsed >= self.goal_timeout:
                raise RunError(f'waypoint simulation timeout after {elapsed:.2f} s')
            self.sleep(self.poll)
        raise RunError(f'waypoint wall-clock timeout after {self.wall_timeout:g} s')

    def run(self, backend: str, scenario: str, route: list[list[float]], run_id: str) -> dict:
        result: dict = {'backend': backend, 'scenario': scenario, 'success': False, 'waypoints': []}
        start = None
        try:
            initial = self.prepare(backend, scenario)
            result['episode_id'] = initial['state']['episode_id']
            start = measurements(initial)
            result['start'] = start
            for index, target in enumerate(route):
                plan_id = f'nav-comparison-{run_id}-{backend}-{index}'
                try:
                    record = self.goal(backend, result['episode_id'], target, plan_id)
                except RunError as error:
                    failed = {'plan_id': plan_id, 'target': target, 'success': False, 'error': str(error)}
                    try:
                        pose = measurements(self.latest)['pose']
                        failed['goal_error_m'] = round(math.hypot(pose['x'] - target[0], pose['y'] - target[1]), 4)
                    except RunError:
                        pass
                    result['waypoints'].append(failed)
                    raise
                result['waypoints'].append(record)
            result['success'] = True
        except RunError as error:
            result['error'] = str(error)
        except KeyboardInterrupt:
            result['error'] = 'interrupted by operator'
            result['interrupted'] = True
        finally:
            # Also runs for Ctrl-C and unexpected exceptions. If stop cannot be
            # confirmed, do not continue to the second backend.
            try:
                stopped = self.stop()
                result['stop_confirmed'] = True
                if start is not None and stopped.get('state', {}).get('episode_id') == result['episode_id']:
                    result['end'] = measurements(stopped)
                    result['metrics'] = metric_delta(start, result['end'])
            except RunError as error:
                result['success'] = False
                result['stop_confirmed'] = False
                result['stop_error'] = str(error)
        return result


def positive(text: str) -> float:
    value = float(text)
    if not math.isfinite(value) or value <= 0:
        raise argparse.ArgumentTypeError('must be a positive finite number')
    return value


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--api', default='http://127.0.0.1:8082')
    parser.add_argument('--backend', choices=('custom', 'nav2', 'both'), default='both')
    parser.add_argument('--scenario', default='easy', help='easy/medium/hard, optionally @seed')
    parser.add_argument('--seed', type=int, help='append an explicit seed to --scenario')
    parser.add_argument('--waypoints', required=True, type=Path, help='JSON file: [[x,y], ...] in odom metres')
    parser.add_argument('--output', required=True, type=Path, help='new JSON result file; existing files are preserved')
    parser.add_argument('--llm-disabled', action='store_true', required=True,
                        help='confirm the stack was started with LLM=false (HTTP cannot verify process launch)')
    parser.add_argument('--poll', type=positive, default=0.2)
    parser.add_argument('--ready-timeout', type=positive, default=45.0)
    parser.add_argument('--goal-timeout', type=positive, default=90.0, help='simulation seconds per waypoint')
    parser.add_argument('--wall-timeout', type=positive, default=180.0, help='wall seconds per waypoint')
    parser.add_argument('--goal-tolerance', type=positive, default=0.20, help='maximum judge world-pose error, metres')
    args = parser.parse_args(argv)
    scenario = args.scenario
    if args.seed is not None:
        if '@' in scenario or args.seed < 0:
            parser.error('--seed requires an unseeded scenario and a nonnegative integer')
        scenario += f'@{args.seed}'
    if not re.fullmatch(r'(easy|medium|hard)(@\d+)?', scenario):
        parser.error('scenario must be easy, medium or hard, optionally @nonnegative_seed')
    try:
        route = route_from_json(args.waypoints.read_text(encoding='utf-8'))
    except (OSError, ValueError) as error:
        parser.error(str(error))
    result = {'schema': 'navigation-comparison@1', 'created_at': datetime.now(timezone.utc).isoformat(),
              'run_id': uuid4().hex[:12], 'scenario': scenario, 'route': route,
              'llm_disabled_asserted': True, 'goal_tolerance_m': args.goal_tolerance,
              'limits': {'goal_simulation_s': args.goal_timeout, 'goal_wall_s': args.wall_timeout,
                         'readiness_wall_s': args.ready_timeout, 'poll_wall_s': args.poll}, 'runs': []}
    try:
        output = args.output.open('x', encoding='utf-8')
    except OSError as error:
        parser.error(f'cannot create output: {error}')
    comparison = Comparison(API(args.api), poll=args.poll, ready_timeout=args.ready_timeout,
                            goal_timeout=args.goal_timeout, wall_timeout=args.wall_timeout,
                            tolerance=args.goal_tolerance)
    try:
        for backend in ('custom', 'nav2') if args.backend == 'both' else (args.backend,):
            run = comparison.run(backend, scenario, route, result['run_id'])
            result['runs'].append(run)
            print(f'{backend}: {"success" if run["success"] else run.get("error", "failed")}', flush=True)
            if not run.get('stop_confirmed') or run.get('interrupted'):
                break
    except (KeyboardInterrupt, RunError) as error:
        result['error'] = 'interrupted by operator' if isinstance(error, KeyboardInterrupt) else str(error)
    finally:
        result['success'] = bool(result['runs']) and all(run['success'] for run in result['runs']) and 'error' not in result
        json.dump(result, output, ensure_ascii=False, indent=2, allow_nan=False)
        output.write('\n')
        output.close()
    print(str(args.output.resolve()))
    return 0 if result['success'] else 1


if __name__ == '__main__':
    sys.exit(main())
