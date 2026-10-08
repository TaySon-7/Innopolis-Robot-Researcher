from math import hypot
from pathlib import Path

import pytest

from did_agent.autonomous import AutonomousAgent
from did_agent.bench import run_scenario
from did_agent.executor import PlanExecutor
from did_agent.plan import parse_plan
from did_agent.search import SampleSearch
from did_agent.sim_robot import SimRobot
from did_agent.skills import Skills
from did_judge.scenario import load_scenario

SCENARIOS = Path(__file__).resolve().parents[2] / 'did_judge' / 'scenarios'


def robot_for(name='easy', **kwargs):
    return SimRobot(load_scenario(SCENARIOS / f'{name}.yaml'), **kwargs)


def test_search_climbs_the_signal_to_a_sample():
    robot = robot_for()
    Skills(robot).goto(-1.5, -1.0)
    result = SampleSearch(robot).run(-1.5, -1.0, 0.5)
    assert result.found
    assert result.peak > 0.8
    # the sample s1 is at (-0.55, -0.55)
    assert hypot(result.x + 0.55, result.y + 0.55) < 0.35
    assert robot.judge.collisions == 0


def test_search_reports_silence_when_nothing_is_near():
    robot = robot_for()
    result = SampleSearch(robot).run(-2.0, -0.5, 0.3)
    # the base is 1.5 m from the nearest sample: out of the sensor's range
    assert not result.found
    assert result.reason == 'no signal'


def test_executor_runs_a_plan_end_to_end():
    robot = robot_for()
    statuses = []
    executor = PlanExecutor(Skills(robot), statuses.append)
    plan = parse_plan(
        '{"plan_id":"t1","subgoals":['
        '{"type":"goto","x":-1.5,"y":-1.0},'
        '{"type":"search_around","x":-1.5,"y":-1.0,"radius":0.5},'
        '{"type":"collect"},'
        '{"type":"return_to_base"}]}'
    )
    final = executor.run(plan)
    assert final['state'] == 'done'
    assert robot.collected() == 1
    assert robot.judge.finished
    assert [s['state'] for s in statuses if s['index'] == 2] == ['running', 'done']
    assert all(s['plan_id'] == 't1' for s in statuses)


def test_executor_stops_at_the_first_failure_and_explains():
    robot = robot_for()
    executor = PlanExecutor(Skills(robot))
    plan = parse_plan(
        '{"subgoals":[{"type":"goto","x":8,"y":8},{"type":"collect"}]}'
    )
    final = executor.run(plan)
    assert final['state'] == 'failed'
    assert final['index'] == 0
    assert 'no path' in final['reason']
    assert robot.collected() == 0 and robot.judge.false_collects == 0


def test_executor_reports_a_false_collect():
    robot = robot_for()
    final = PlanExecutor(Skills(robot)).run(parse_plan('[{"type":"collect"}]'))
    assert final['state'] == 'failed'
    assert 'no sample' in final['reason']


def test_executor_stops_when_preempted():
    robot = robot_for()
    robot.on_tick = lambda r: setattr(r, 'preempt', r.now() > 3.0)
    final = PlanExecutor(Skills(robot)).run(
        parse_plan('[{"type":"goto","x":1.7,"y":-0.5},{"type":"collect"}]')
    )
    assert final['state'] == 'preempted'
    assert robot.judge.false_collects == 0


def test_return_to_base_cost_estimate_is_sane():
    robot = robot_for()
    skills = Skills(robot)
    assert skills.return_cost_estimate() < 0.3
    skills.goto(1.7, 0.5)
    estimate = skills.return_cost_estimate()
    assert 3.5 < estimate < 8.0


def test_agent_never_strands_itself_with_a_tiny_battery():
    scenario = load_scenario(SCENARIOS / 'hard.yaml')
    scenario.initial_battery = 12.0
    robot = SimRobot(scenario)
    summary = AutonomousAgent(robot).run()
    assert summary['returned_to_base']
    assert robot.battery() > 0.0


@pytest.mark.parametrize(
    ('name', 'minimum', 'false_collects'),
    # hard has a sensor fault (noise 0.15) that makes a few false collects likely
    [('easy', 3, 1), ('medium', 5, 1), ('hard', 6, 4)],
)
def test_autonomous_agent_collects_and_comes_home(name, minimum, false_collects):
    summary = run_scenario(name)
    assert summary['collected'] >= minimum
    assert summary['returned_to_base'] and summary['finished']
    assert summary['collisions'] == 0
    assert summary['false_collects'] <= false_collects
    assert summary['battery'] > 0.0


def test_agent_copes_with_wheel_slip():
    robot = robot_for('easy', yaw_noise=0.02)
    summary = AutonomousAgent(robot).run()
    assert summary['collected'] >= 2
    assert summary['returned_to_base']
    assert robot.sim.collisions == 0


def test_bench_expands_seed_ranges_and_summarizes():
    from did_agent.bench import expand
    from did_agent.bench import summarize
    assert expand('hard@1-3') == ['hard@1', 'hard@2', 'hard@3']
    assert expand('hard@7') == ['hard@7'] and expand('easy') == ['easy']
    rows = [run_scenario(name) for name in expand('easy@1-2')]
    summary = summarize('easy@1-2', rows)
    assert summary['episodes'] == 2 and summary['all_returned']
    assert summary['collected'] == summary['samples'] == 6
