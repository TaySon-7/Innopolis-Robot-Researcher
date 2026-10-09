"""Run the LLM planner against the real agent, without Gazebo.

This is the comparison the handbook asks for under "ideas for later": the same
scenarios, the same judge, the same skills, the same plan parser — the only
difference is who writes the plan.

``did_agent`` already ships a kinematic simulator whose ``SimRobot`` implements
the same ``Robot`` interface as the ROS node. That means the planner can be
exercised against real scenarios, a real cost map, the real search and the
real failure reasons, in seconds, with nothing simulated twice.

The gap against the stand is stated plainly: no Gazebo, no lidar noise, no
wheel slip, no contacts, and no ROS transport. Conclusions about the planner's
reasoning transfer; conclusions about driving do not.

    python3 -m did_llm.integration easy medium hard
    python3 -m did_llm.integration hard@1-8 --compare
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from typing import Any, Callable

from did_agent.executor import PlanExecutor
from did_agent.plan import PlanError, parse_plan
from did_agent.scenario_generator import load_named
from did_agent.sim_robot import SimRobot
from did_agent.skills import Skills

from did_llm.agent_plan import PlanRejected, parse_model_plan
from did_llm.llm_client import (
    LLMClient,
    LLMConfig,
    LLMUnavailable,
    load_api_key,
)
from did_llm.planner_node import Planner, PlannerConfig
from did_llm.prompts import PLANNER_SYSTEM, build_planner_prompt


class RobotLink:
    """The planner's view of a robot, without ROS.

    Implements the same surface as :class:`did_llm.agent_link.AgentLink`, so
    :class:`~did_llm.planner_node.Planner` drives it unchanged. Duck typing is
    the point: the planning loop has no idea whether it is talking to a
    simulator or to a live agent, which is what makes the comparison fair.
    """

    def __init__(self, robot: SimRobot, skills: Skills,
                 log: Callable[[str], None]) -> None:
        self.robot = robot
        self.skills = skills
        self.log = _Log(log)
        self.state: dict[str, Any] | None = None
        self.status: dict[str, Any] | None = None
        self.episode_finished = False
        self.state_at: float = 0.0
        self.clock = 0.0
        self.autonomous = False

        self.published: list[dict[str, Any]] = []
        self.commands: list[str] = []
        self.journal_entries: list[dict[str, Any]] = []

        self.refresh()

    # ------------------------------------------------------------------ input
    def refresh(self) -> None:
        """Rebuild /agent/state from the robot, in the executor's format."""
        pose = self.robot.pose()
        try:
            return_cost = self.skills.return_cost_estimate()
        except Exception:  # noqa: BLE001 - mirrors the agent's own guard
            return_cost = float('nan')
        reading = self.robot.read_sensor()

        self.state = {
            't': round(self.robot.now(), 2),
            'scenario': self.robot.judge.scenario.name,
            'pose': {'x': round(pose.x, 3), 'y': round(pose.y, 3),
                     'yaw': round(pose.yaw, 3)},
            'battery': round(self.robot.battery(), 3),
            'collected': self.robot.collected(),
            'samples_total': self.robot.samples_total(),
            'score': round(self.robot.judge.score, 1),
            'sensor': {'value': round(reading.value, 3),
                       'noise_estimate': round(self.robot.noise_level(), 4)},
            'current': {'plan_id': (self.status or {}).get('plan_id', ''),
                        'index': (self.status or {}).get('index', 0),
                        'type': (self.status or {}).get('type', ''),
                        'state': (self.status or {}).get('state', 'idle')},
            'recent_events': [],
            'return_cost_estimate': (None if return_cost != return_cost
                                     else round(return_cost, 2)),
            'anomaly': self.robot.anomaly(),
            'cost_map_updates': [],
            'navigation': {'status': 'idle', 'replans': 0},
        }
        self.state_at = self.now()
        self.episode_finished = bool(self.robot.judge.finished)
        # The planner measures time to decide when the next plan is due, so
        # the clock has to move with the simulation. A frozen clock would make
        # every plan look fresh and the planner would stop after the first one.
        self.clock = self.robot.now()

    # ----------------------------------------------------------------- output
    def now(self) -> float:
        return self.clock

    def advance_clock(self, seconds: float) -> None:
        self.clock += seconds

    def state_age(self) -> float:
        return self.now() - self.state_at

    def battery(self) -> float:
        return float((self.state or {}).get('battery', 0.0))

    def return_cost(self) -> float | None:
        value = (self.state or {}).get('return_cost_estimate')
        return float(value) if isinstance(value, (int, float)) else None

    def remaining_samples(self) -> int:
        state = self.state or {}
        return max(0, int(state.get('samples_total', 0))
                   - int(state.get('collected', 0)))

    def finished(self) -> bool:
        return self.episode_finished

    def in_autonomous_mode(self) -> bool:
        """True once the harness has handed the episode to the agent's policy.

        The live agent reports this as ``plan_id: "auto"``; here the harness
        knows because it is the one that started it.
        """
        return self.autonomous

    def publish_plan(self, payload: dict[str, Any]) -> None:
        self.published.append(payload)

    def publish_command(self, command: str) -> None:
        self.commands.append(command)

    def journal(self, kind: str, title: str, text: str = '',
                status: str = 'open', **extra: Any) -> None:
        self.journal_entries.append({'kind': kind, 'title': title,
                                     'text': text, 'status': status, **extra})
        # The live dashboard renders these; the harness has no dashboard, so
        # without this a plan's refusal reason would only ever appear in a
        # container log and never in the comparison output.
        self.log.info(f'[{kind}/{status}] {title}'
                    + (f' — {text}' if text else ''))

    def last_status_for(self, plan_id: str) -> dict[str, Any] | None:
        status = self.status
        if not status or status.get('plan_id') != plan_id:
            return None
        if status.get('state') not in ('done', 'failed', 'preempted'):
            return None
        return status


class _Log:
    """Minimal stand-in for a ROS logger."""

    def __init__(self, sink: Callable[[str], None]) -> None:
        self.sink = sink

    def info(self, message: str) -> None:
        self.sink(message)

    def warn(self, message: str) -> None:
        self.sink(f'ВНИМАНИЕ: {message}')

    def error(self, message: str) -> None:
        self.sink(f'ОШИБКА: {message}')

    def debug(self, message: str) -> None:
        pass


def run_llm(scenario: str, client: LLMClient, *, max_plans: int = 12,
            verbose: bool = True) -> dict[str, Any]:
    """One episode with the LLM in charge of the plan."""
    robot = SimRobot(load_named(scenario))
    # The executor needs Skills, which the ROS node builds around its Robot;
    # SimRobot is the Robot itself, so the same wrapper goes around it.
    skills = Skills(robot)
    lines: list[str] = []

    def say(message: str) -> None:
        lines.append(message)
        if verbose:
            print(f'    {message}', flush=True)

    link = RobotLink(robot, skills, say)
    planner = Planner(link, client, PlannerConfig())

    executor = PlanExecutor(
        skills=skills,
        publish_status=lambda status: setattr(link, 'status', status),
    )

    started = time.time()
    plans = 0
    while plans < max_plans and not robot.judge.finished:
        link.refresh()
        before = len(link.published)
        planner.tick()

        # Run whatever the planner decided, through the real executor.
        while len(link.published) > before:
            payload = link.published[before]
            before += 1
            plans += 1
            try:
                parsed = parse_plan(json.dumps(payload))
            except PlanError as error:
                # The planner is supposed to catch this itself; if it slips
                # through, the rejection has to reach it the same way the
                # agent's would.
                link.status = {'plan_id': payload.get('plan_id', ''),
                               'index': 0, 'type': '', 'subgoal': '',
                               'state': 'failed',
                               'reason': f'invalid plan: {error}', 'data': {}}
                say(f'ИСПОЛНИТЕЛЬ ОТКЛОНИЛ ПЛАН: {error}')
                continue

            say(f'план {parsed.plan_id}: '
                f'{[s.describe() for s in parsed.subgoals]}')
            result = executor.run(parsed)
            if result.get('state') == 'failed':
                say(f'  подцель не выполнена: {result.get("reason", "")}')

        if planner.handed_over:
            say('планировщик передал эпизод автономному режиму')
            _drive_autonomous(robot, executor, link, say)
            break
        link.advance_clock(0.0)

    if not robot.judge.finished and not planner.handed_over:
        # Let the agent finish on its own so the episode still ends properly.
        _drive_autonomous(robot, executor, link, say)

    judge = robot.judge
    return {
        'scenario': scenario,
        'planner': 'llm',
        'collected': judge.collected_count,
        'samples_total': len(judge.samples),
        'finished': judge.finished,
        'battery': round(judge.battery, 1),
        'distance': round(robot.sim.distance, 2),
        'score': round(judge.score, 1),
        'collisions': judge.collisions,
        'hazard_hits': judge.hazard_hits,
        'false_collects': judge.false_collects,
        'plans': plans,
        'wall_seconds': round(time.time() - started, 1),
        'hypotheses': robot.adaptation.hypotheses,
    }


def _drive_autonomous(robot: SimRobot, executor: PlanExecutor,
                      link: 'RobotLink', say: Callable[[str], None]) -> None:
    """Finish the episode with the agent's own policy."""
    from did_agent.autonomous import AutonomousAgent

    say('автономный режим')
    link.autonomous = True
    AutonomousAgent(robot, log=say).run()


def run_autonomous(scenario: str) -> dict[str, Any]:
    """The baseline: the same episode with nobody planning."""
    from did_agent.autonomous import AutonomousAgent

    robot = SimRobot(load_named(scenario))
    started = time.time()
    AutonomousAgent(robot).run()
    judge = robot.judge
    return {
        'scenario': scenario,
        'planner': 'autonomous',
        'collected': judge.collected_count,
        'samples_total': len(judge.samples),
        'finished': judge.finished,
        'battery': round(judge.battery, 1),
        'distance': round(robot.sim.distance, 2),
        'score': round(judge.score, 1),
        'collisions': judge.collisions,
        'hazard_hits': judge.hazard_hits,
        'false_collects': judge.false_collects,
        'plans': 0,
        'wall_seconds': round(time.time() - started, 1),
        'hypotheses': robot.adaptation.hypotheses,
    }


def expand(spec: str) -> list[str]:
    """``easy@1-8`` into a list; single names pass through."""
    base, _, seeds = spec.partition('@')
    if not seeds:
        return [spec]
    first, dash, last = seeds.partition('-')
    if dash and first.isdigit() and last.isdigit():
        return [f'{base}@{seed}' for seed in range(int(first), int(last) + 1)]
    return [f'{base}@{seeds}']


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('scenarios', nargs='+',
                        help='easy medium hard hard@7 easy@1-8')
    parser.add_argument('--compare', action='store_true',
                        help='Also run the autonomous agent on the same scenarios.')
    parser.add_argument('--max-plans', type=int, default=12)
    parser.add_argument('--base-url', default='https://api-ai.mai.ru/v1')
    parser.add_argument('--model', default='deepseek-v4.1-flash')
    args = parser.parse_args(argv)

    client = LLMClient(LLMConfig(base_url=args.base_url,
                                 api_key=load_api_key(),
                                 model=args.model))
    if not client.cfg.configured:
        print('НЕТ КЛЮЧА API — сравнение с LLM невозможно.', file=sys.stderr)
        return 2

    names: list[str] = []
    for spec in args.scenarios:
        names.extend(expand(spec))

    rows = []
    for name in names:
        print(f'\n=== {name}: LLM-планировщик ===', flush=True)
        try:
            rows.append(run_llm(name, client, max_plans=args.max_plans))
        except (LLMUnavailable, PlanRejected) as error:
            print(f'  пропущено: {error}')
            continue
        if args.compare:
            print(f'=== {name}: автономный агент ===', flush=True)
            rows.append(run_autonomous(name))

    print('\n' + '=' * 92)
    print(f'{"сценарий":<14}{"планировщик":<14}{"собрано":>9}{"вернулся":>10}'
          f'{"батарея":>9}{"дистанция":>11}{"штрафы":>8}{"планов":>8}')
    print('-' * 92)
    for row in rows:
        print(f'{row["scenario"]:<14}{row["planner"]:<14}'
              f'{str(row["collected"]) + "/" + str(row["samples_total"]):>9}'
              f'{"да" if row["finished"] else "НЕТ":>10}'
              f'{row["battery"]:>9}{row["distance"]:>11}'
              f'{row["collisions"] + row["hazard_hits"]:>8}{row["plans"]:>8}')
    print('=' * 92)

    if args.compare:
        _print_verdict(rows)
    return 0


def _print_verdict(rows: list[dict[str, Any]]) -> None:
    """The one number worth reporting: does the model beat the policy?"""
    llm = [r for r in rows if r['planner'] == 'llm']
    auto = [r for r in rows if r['planner'] == 'autonomous']
    if not llm or not auto:
        return
    def mean(items, key):
        return sum(i[key] for i in items) / len(items)
    print(f'\nLLM: собрано {mean(llm, "collected"):.2f}, '
          f'вернулось {sum(1 for r in llm if r["finished"])}/{len(llm)}, '
          f'штрафов {mean(llm, "collisions") + mean(llm, "hazard_hits"):.2f}')
    print(f'Правила: собрано {mean(auto, "collected"):.2f}, '
          f'вернулось {sum(1 for r in auto if r["finished"])}/{len(auto)}, '
          f'штрафов {mean(auto, "collisions") + mean(auto, "hazard_hits"):.2f}')


if __name__ == '__main__':
    raise SystemExit(main())
