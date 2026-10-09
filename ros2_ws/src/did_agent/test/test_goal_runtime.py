"""ROS-adapter lifecycle tests without launching nodes or an external model."""

import json
from types import SimpleNamespace

import numpy as np

from did_agent.agent_node import AgentNode
from did_agent.controller import Pose
from did_agent.costmap import CostMap
from did_agent.executor import PlanExecutor
from did_agent.experimental_goals import BudgetPolicy, Observation
from did_agent.goal_offers import GoalOffers
from did_agent.grid import GridMap
from did_agent.nav_node import Navigator
from did_agent.plan import parse_plan
from did_agent.robot import Reading
from did_agent.search import SampleSearch
from did_agent.skills import SkillResult


def test_goto_never_publishes_motion_after_preemption_clears_pose(monkeypatch):
    valid_pose = Pose(0.2, 0.3, 0.0)
    state = {'pose': valid_pose, 'preempted': False, 'stops': 0}
    core = SimpleNamespace(set_goal=lambda *_: True, cancel=lambda: None,
                           update=lambda *_: (_ for _ in ()).throw(AssertionError('moved after stop')))
    nav = SimpleNamespace(
        core=core, ready=lambda: True, wait_for_sensors=lambda: True,
        pose=lambda: state['pose'], fresh_pose=lambda: state['pose'], now=lambda: 0.0,
        preempted=lambda: state['preempted'],
        stop=lambda: state.update(stops=state['stops'] + 1),
        _result=lambda pose: {'pose': {'x': pose.x, 'y': pose.y}},
    )

    def preempt(*_args, **_kwargs):
        state.update(pose=None, preempted=True)

    monkeypatch.setattr('did_agent.nav_node.rclpy.ok', lambda: True)
    monkeypatch.setattr('did_agent.nav_node.rclpy.spin_once', preempt)
    assert Navigator.goto(nav, 1, 1)['reason'] == 'preempted'
    assert state['stops'] == 1


def test_search_returns_clean_preemption_when_reset_interrupts_sensor_read():
    state = {'preempted': False, 'pose': Pose(0, 0, 0)}

    def read(_count):
        state.update(preempted=True, pose=None)
        return Reading(0.8, 0.0)

    robot = SimpleNamespace(preempted=lambda: state['preempted'], pose=lambda: state['pose'],
                            read_sensor=read, noise_level=lambda: 0.0)
    result = SampleSearch(robot).run(0, 0, 0.5)
    assert not result.found and result.reason == 'preempted'


def test_math_executor_guards_goto_and_only_final_status_completes_plan():
    goto_calls, statuses = [], []
    skills = SimpleNamespace(
        goto=lambda x, y, guarded=False, stop_on_signal=False: (
            goto_calls.append((guarded, stop_on_signal)) or SkillResult(True)
        ),
        collect=lambda: SkillResult(True), robot=SimpleNamespace(preempted=lambda: False),
    )
    plan = parse_plan('{"plan_id":"m","subgoals":[{"type":"goto","x":1,"y":1},{"type":"collect"}]}')
    PlanExecutor(skills, statuses.append).run(plan, guarded=True)
    assert goto_calls == [(True, True)]
    assert [s['plan_complete'] for s in statuses] == [False, False, False, True]


def manager_fixture():
    grid = GridMap(0.25, 0, 0, np.zeros((16, 16), bool), np.zeros((16, 16), bool))
    observation = Observation('episode', 0, (0.625, 0.625), 100, samples_total=3)
    return GoalOffers(CostMap(grid), observation.pose), observation


def test_rejected_idle_selection_gets_new_token_without_changing_execution_status():
    manager, observation = manager_fixture()
    old = manager.build(observation)
    sent = []
    node = SimpleNamespace(goal_offers=manager, _current={'plan_id': 'unchanged'},
                           get_logger=lambda: SimpleNamespace(warning=lambda *_: None),
                           _status_pub=SimpleNamespace(publish=lambda msg: sent.append(json.loads(msg.data))))
    AgentNode._reject_selection(node, {'plan_id': 'bad', 'goal_selection': {
        'snapshot_id': old['snapshot_id'], 'goal_id': 'invented'}}, 'unknown goal')
    assert manager.build(observation)['snapshot_id'] != old['snapshot_id']
    assert node._current == {'plan_id': 'unchanged'}
    assert sent[0]['plan_id'] == 'bad' and sent[0]['data']['accepted'] is False


def test_late_llm_answer_cannot_interrupt_running_search():
    rejected = []
    node = SimpleNamespace(_control_mode='llm', _executing=True,
                           _pending=None, _pending_selection=None, _preempt=False,
                           _reject_selection=lambda envelope, reason: rejected.append(reason))
    AgentNode._on_plan(node, SimpleNamespace(data=json.dumps({
        'plan_id': 'late', 'goal_selection': {'snapshot_id': 'old', 'goal_id': 'A'}})))
    assert rejected and not node._preempt and node._pending_selection is None


def test_stop_revokes_offer_and_llm_requires_explicit_resume():
    manager, observation = manager_fixture()
    manager.build(observation)
    stops = []
    node = SimpleNamespace(goal_offers=manager, _pending=None,
                           get_logger=lambda: SimpleNamespace(info=lambda *_: None),
                           core=SimpleNamespace(cancel=lambda: None), stop=lambda: stops.append(True),
                           journal=lambda *_: None)
    AgentNode._on_command(node, SimpleNamespace(data='{"cmd":"stop"}'))
    assert node._control_mode == 'stopped' and node._preempt and manager.offer is None
    AgentNode._on_command(node, SimpleNamespace(data='{"cmd":"llm"}'))
    assert node._control_mode == 'llm' and len(stops) == 2


def test_scenario_switch_keeps_pose_until_old_skill_unwinds_then_clears_it():
    manager, _ = manager_fixture()
    node = SimpleNamespace(goal_offers=manager, world_pose=object(), _episode_id=0, cost_log=[],
                           get_logger=lambda: SimpleNamespace(info=lambda *_: None),
                           core=SimpleNamespace(cancel=lambda: None), stop=lambda: None,
                           journal=lambda *_: None, _publish_status=lambda *_: None,
                           _publish_costmap=lambda: None, _budget_policy=lambda: BudgetPolicy())
    node.clear_world_pose = lambda: setattr(node, 'world_pose', None)
    pose = node.world_pose
    AgentNode._on_scenario_select(node, SimpleNamespace(data='easy'))
    assert node.world_pose is pose and node._preempt
    AgentNode._reset_for_scenario(node, 'easy')
    assert node.world_pose is None and node._episode_id == 1
