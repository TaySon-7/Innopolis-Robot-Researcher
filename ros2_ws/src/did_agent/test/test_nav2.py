"""Observed-map parity and the Nav2 action/velocity ownership boundary."""

from math import pi
import time
from types import SimpleNamespace

import numpy as np
import pytest

from did_agent.controller import Command, Pose
from did_agent.costmap import CostMap
from did_agent.grid import GridMap
from did_agent.nav2_core import checked_velocity, knowledge_grid
from did_agent.navigator_core import DONE, FAILED, IDLE, RUNNING, Scan


def small_costmap():
    occupied = np.zeros((5, 6), dtype=bool)
    unknown = np.zeros_like(occupied)
    occupied[0, 0] = True
    unknown[4, 5] = True
    grid = GridMap(1.0, -3.0, -2.0, occupied, unknown)
    return CostMap(grid, robot_radius=1.1, inflation_radius=2.5)


def full_scan(*hits, stamp=10.0):
    ranges = np.full(360, np.inf)
    for angle, distance in hits:
        ranges[round(angle * 180 / pi) % 360] = distance
    return Scan(0.0, pi / 180, ranges, stamp=stamp)


def test_knowledge_grid_preserves_world_row_order_and_unknown_space():
    costmap = small_costmap()
    costmap.update({'cells': [[1, 4]]}, 3.0)

    result = knowledge_grid(costmap)

    assert result.shape == (5, 6)
    assert result.dtype == np.int8
    assert result[0, 0] == 100  # the wall is at the lowest world y
    assert result[4, 5] == -1
    row, col = costmap.world_to_cell(*costmap.cell_to_world(1, 4))
    assert result[row, col] > 0
    assert result.ravel()[1 * 6 + 4] == result[row, col]
    assert result[3, 4] == 0  # no accidental vertical image flip


def test_nav2_inflates_raw_walls_only_once_but_keeps_learned_blockers():
    costmap = small_costmap()
    assert costmap.static_blocked[0, 1]
    assert not costmap.grid.occupied[0, 1]
    assert costmap.wall_cost[0, 1] > 0
    x, y = costmap.cell_to_world(2, 2)
    costmap.block_disk(x, y, 0.1)

    result = knowledge_grid(costmap)

    assert result[0, 0] == 100
    assert result[0, 1] == 0  # Nav2 supplies its own footprint/inflation
    assert result[2, 2] == 100  # observed dynamic exclusion must survive


def test_nav2_terrain_costs_increase_without_becoming_impassable():
    costmap = small_costmap()
    costs = [0.5, 1.0, 2.0, 4.0, 10.0, 1000.0]
    costmap.terrain[2, :] = costs

    result = knowledge_grid(costmap)[2]

    assert result[0] == result[1] == 0
    assert result[1] < result[2] < result[3] < result[4]
    assert result[4] == result[5]
    assert np.all((0 <= result) & (result < 100))


@pytest.mark.parametrize('value', [float('nan'), float('inf'), -float('inf'), 0.0, -1.0])
def test_invalid_observed_costs_cannot_enter_nav2_map(value):
    costmap = small_costmap()
    costmap.terrain[2, 2] = value

    with pytest.raises(ValueError, match='terrain'):
        knowledge_grid(costmap)


@pytest.mark.parametrize(('linear', 'angular', 'expected'), [
    (5.0, 5.0, Command(0.18, 1.2)),
    (-5.0, -5.0, Command(-0.18, -1.2)),
    (0.1, -0.5, Command(0.1, -0.5)),
])
def test_nav2_commands_are_bounded_at_robot_limits(linear, angular, expected):
    command, reason = checked_velocity(Command(linear, angular), full_scan(), 10.0)

    assert command == expected
    assert reason == ''


@pytest.mark.parametrize(('command', 'now'), [
    (Command(float('nan'), 0.0), 10.0),
    (Command(0.1, float('inf')), 10.0),
    (Command(0.1, 0.0), float('nan')),
])
def test_nonfinite_nav2_commands_stop(command, now):
    checked, reason = checked_velocity(command, full_scan(), now)

    assert checked == Command()
    assert 'invalid' in reason


@pytest.mark.parametrize('scan', [
    None, full_scan(stamp=None), full_scan(stamp=9.0),
    full_scan(stamp=10.2), full_scan(stamp=float('nan')),
    Scan(0.0, pi / 180, np.full(360, np.nan), stamp=10.0),
    Scan(0.0, pi / 180, np.array([]), stamp=10.0),
])
def test_blind_stale_or_future_scans_stop_nav2(scan):
    command, reason = checked_velocity(Command(0.1, 0.0), scan, 10.0)

    assert command == Command()
    assert reason


@pytest.mark.parametrize(('field', 'value'), [
    (field, value)
    for field in ('angle_min', 'angle_increment', 'range_min')
    for value in (float('nan'), float('inf'), -float('inf'))
] + [('angle_increment', 0.0), ('range_min', -0.01)])
def test_invalid_scan_geometry_blocks_turning_even_with_clear_ranges(field, value):
    scan = full_scan()
    setattr(scan, field, value)

    command, reason = checked_velocity(Command(0.0, 0.5), scan, 10.0)

    assert command == Command()
    assert reason == 'invalid lidar scan geometry'


@pytest.mark.parametrize(('field', 'value'), [('angle_increment', -pi / 180), ('range_min', 0.0)])
def test_valid_scan_geometry_still_allows_turning(field, value):
    scan = full_scan()
    setattr(scan, field, value)

    command, reason = checked_velocity(Command(0.0, 0.5), scan, 10.0)

    assert command == Command(0.0, 0.5)
    assert reason == ''


@pytest.mark.parametrize('distance', [float('nan'), -float('inf'), 0.0, 0.11])
def test_invalid_beam_in_motion_direction_is_not_clear_space(distance):
    command, reason = checked_velocity(Command(0.1), full_scan((0.0, distance)), 10.0)

    assert command == Command()
    assert 'coverage' in reason


@pytest.mark.parametrize(('linear', 'angle'), [(0.1, 0.0), (-0.1, pi)])
def test_emergency_stop_checks_the_actual_direction_of_motion(linear, angle):
    command, reason = checked_velocity(Command(linear), full_scan((angle, 0.12)), 10.0)

    assert command == Command()
    assert 'emergency' in reason


def test_reversing_requires_rear_coverage():
    front_only = Scan(-0.3, 0.1, np.full(7, np.inf), stamp=10.0)

    command, reason = checked_velocity(Command(-0.1), front_only, 10.0)

    assert command == Command()
    assert 'coverage' in reason


def test_turning_checks_rear_clearance_even_without_translation():
    command, reason = checked_velocity(Command(0.0, 0.5), full_scan((pi, 0.12)), 10.0)

    assert command == Command()
    assert 'turning clearance' in reason


def test_turning_requires_all_scan_beams_to_be_usable():
    command, reason = checked_velocity(Command(0.0, 0.5), full_scan((pi, np.nan)), 10.0)

    assert command == Command()
    assert 'coverage for turning' in reason


class Future:
    """Only the asynchronous action surface used by the adapter."""

    def __init__(self, result=None, *, done=True, error=None):
        self.value = result
        self.complete = done
        self.error = error
        self.cancel_count = 0

    def done(self):
        return self.complete

    def exception(self):
        return self.error

    def result(self):
        if self.error is not None:
            raise self.error
        return self.value

    def add_done_callback(self, callback):
        if self.complete:
            callback(self)

    def cancel(self):
        self.cancel_count += 1
        self.complete = True


class GoalHandle:
    def __init__(self, accepted=True):
        self.accepted = accepted
        self.cancel_count = 0
        self.result_future = Future(done=False)

    def get_result_async(self):
        return self.result_future

    def cancel_goal_async(self):
        self.cancel_count += 1
        return Future()


@pytest.fixture
def nav2_module():
    pytest.importorskip('rclpy')
    from did_agent import nav2_adapter
    return nav2_adapter


@pytest.fixture
def adapter(nav2_module):
    # No node/executor/servers: callbacks still run the real ownership logic.
    nav = object.__new__(nav2_module.Nav2Adapter)
    nav.status, nav.reason = IDLE, ''
    nav.goal = None
    nav.waypoints = []
    nav.replans = nav.recoveries = 0
    nav._path_signature = None
    nav._generation = 0
    nav._accepted = False
    nav._started = 10.0
    nav._velocity = None
    nav._requests = []
    nav._clear_futures = []
    nav._lifecycle_active = True
    nav._lifecycle_checked = time.monotonic()
    nav._lifecycle_future = None
    nav._lifecycle_requested = 0.0
    published = []
    nav.node = SimpleNamespace(ready=lambda: True, now=lambda: 10.3,
                               publish=published.append, published=published)
    nav.client = SimpleNamespace(server_is_ready=lambda: True)
    return nav


def velocity_message(stamp=10.0, linear=0.1, angular=0.0):
    sec = int(stamp)
    return SimpleNamespace(
        header=SimpleNamespace(stamp=SimpleNamespace(sec=sec, nanosec=round((stamp - sec) * 1e9))),
        twist=SimpleNamespace(linear=SimpleNamespace(x=linear), angular=SimpleNamespace(z=angular)),
    )


def test_late_goal_acceptance_after_cancel_cannot_reopen_velocity_gate(adapter, nav2_module):
    request = nav2_module._Request(adapter._generation, sent=Future(done=False))
    adapter._requests = [request]
    adapter.status = RUNNING
    adapter.cancel()  # Goal response has not arrived yet.
    handle = GoalHandle()
    accepted = Future(handle)
    request.sent = accepted

    adapter._accepted_goal(request, accepted)
    adapter._on_velocity(velocity_message())

    assert handle.cancel_count == 1
    assert request.cancelled
    assert not adapter._accepted
    assert adapter._velocity is None
    assert adapter.status == FAILED
    assert not request.finished()  # Cancellation is not complete just because it was sent.
    handle.result_future.complete = True
    assert request.finished()


def test_cancellation_discards_buffered_velocity_and_cancels_only_once(adapter, nav2_module):
    handle = GoalHandle()
    request = nav2_module._Request(adapter._generation, sent=Future(handle), handle=handle)
    adapter._requests = [request]
    adapter.status, adapter._accepted = RUNNING, True
    adapter._on_velocity(velocity_message())
    assert adapter._velocity is not None

    adapter.cancel()
    adapter.cancel()
    adapter._on_velocity(velocity_message(stamp=10.1))

    assert handle.cancel_count == 1
    assert adapter._velocity is None
    assert not adapter._accepted
    assert adapter.reason == 'preempted'
    assert adapter.node.published and all(command == Command() for command in adapter.node.published)


@pytest.mark.parametrize(('status', 'accepted', 'stamp'), [
    (IDLE, True, 10.0), (DONE, True, 10.0), (FAILED, True, 10.0),
    (RUNNING, False, 10.0), (RUNNING, True, 9.99),
])
def test_velocity_is_ignored_outside_the_current_accepted_action(adapter, status, accepted, stamp):
    adapter.status, adapter._accepted = status, accepted

    adapter._on_velocity(velocity_message(stamp=stamp))

    assert adapter._velocity is None


def test_current_action_accepts_fresh_velocity(adapter, nav2_module):
    handle = GoalHandle()
    request = nav2_module._Request(adapter._generation)
    adapter.status = RUNNING

    adapter._accepted_goal(request, Future(handle))
    adapter._on_velocity(velocity_message(stamp=10.2, linear=0.12, angular=0.3))

    assert adapter._accepted
    assert adapter._velocity == (Command(0.12, 0.3), pytest.approx(10.2))
    assert handle.cancel_count == 0


def test_action_acceptance_failure_is_visible(adapter, nav2_module):
    adapter.status = RUNNING
    request = nav2_module._Request(adapter._generation)

    adapter._accepted_goal(request, Future(error=RuntimeError('server disconnected')))

    assert adapter.status == FAILED
    assert 'server disconnected' in adapter.reason
    assert not adapter._accepted


def test_readiness_snapshot_exposes_server_loss_and_hides_old_routes(adapter):
    adapter.waypoints = [(0.0, 1.0)]
    adapter.client.server_is_ready = lambda: False

    assert adapter.snapshot()['ready'] is False
    assert 'not ready' in adapter.snapshot()['reason']
    assert adapter.snapshot()['waypoints'] == []
    adapter.client.server_is_ready = lambda: True
    adapter.status, adapter.reason = FAILED, 'goal blocked'
    assert adapter.snapshot()['ready'] is True
    assert adapter.snapshot()['reason'] == 'goal blocked'


@pytest.mark.parametrize('unready', ['inactive', 'stale_lifecycle', 'sensors'])
def test_readiness_needs_active_lifecycle_fresh_health_and_robot_sensors(adapter, unready):
    if unready == 'inactive':
        adapter._lifecycle_active = False
    elif unready == 'stale_lifecycle':
        adapter._lifecycle_checked = time.monotonic() - 6.0
    else:
        adapter.node.ready = lambda: False

    assert adapter.snapshot()['ready'] is False
    assert adapter.snapshot()['reason']


@pytest.mark.parametrize(('state_id', 'active'), [(1, False), (2, False), (3, True), (4, False)])
def test_lifecycle_state_response_controls_readiness(adapter, state_id, active):
    adapter._lifecycle_checked = 0.0
    future = Future(SimpleNamespace(current_state=SimpleNamespace(id=state_id)))
    adapter._lifecycle_future = future

    adapter._lifecycle_result(future)

    assert adapter._lifecycle_active is active
    assert adapter.ready() is active
    assert adapter._lifecycle_checked > 0.0


def test_failed_lifecycle_poll_clears_readiness(adapter):
    future = Future(error=RuntimeError('service disconnected'))
    adapter._lifecycle_future = future
    adapter._lifecycle_result(future)

    assert not adapter.ready()


def test_lost_lifecycle_response_is_retried_and_cannot_override_new_health(
    adapter, nav2_module, monkeypatch,
):
    clock, removed, sent = [10.0], [], []
    lost, replacement = Future(done=False), Future(done=False)
    adapter._lifecycle_future = lost
    adapter._lifecycle_requested = adapter._lifecycle_checked = clock[0]
    adapter._lifecycle_client = SimpleNamespace(
        service_is_ready=lambda: True,
        remove_pending_request=removed.append,
        call_async=lambda request: sent.append(request) or replacement,
    )
    monkeypatch.setattr(nav2_module.time, 'monotonic', lambda: clock[0])

    clock[0] = 11.0
    adapter._check_lifecycle()
    assert adapter.ready()
    assert not sent and not removed

    clock[0] = 12.0
    adapter._check_lifecycle()
    assert removed == [lost]
    assert lost.cancel_count == 1
    assert len(sent) == 1
    assert adapter._lifecycle_future is replacement
    assert not adapter.ready()

    # Even a late active response from the abandoned request cannot reopen
    # readiness while the new server's status remains unknown.
    lost.value = SimpleNamespace(current_state=SimpleNamespace(id=3))
    adapter._lifecycle_result(lost)
    assert not adapter.ready()
    assert adapter._lifecycle_future is replacement

    replacement.value = SimpleNamespace(current_state=SimpleNamespace(id=3))
    replacement.complete = True
    adapter._lifecycle_result(replacement)
    assert adapter.ready()
    assert adapter._lifecycle_future is None
    assert adapter._lifecycle_checked == 12.0


def test_lifecycle_poll_recovers_when_service_returns_after_disconnection(
    adapter, nav2_module, monkeypatch,
):
    clock, available, removed = [20.0], [False], []
    pending = Future(done=False)
    adapter._lifecycle_future = pending
    adapter._lifecycle_requested = 17.0
    adapter._lifecycle_client = SimpleNamespace(
        service_is_ready=lambda: available[0],
        remove_pending_request=removed.append,
        call_async=lambda request: Future(SimpleNamespace(current_state=SimpleNamespace(id=3))),
    )
    monkeypatch.setattr(nav2_module.time, 'monotonic', lambda: clock[0])

    adapter._check_lifecycle()
    assert removed == [pending]
    assert adapter._lifecycle_future is None
    assert not adapter.ready()

    available[0] = True
    clock[0] = 21.0
    adapter._check_lifecycle()
    assert adapter.ready()
    assert adapter._lifecycle_checked == 21.0


def test_lifecycle_request_send_failure_does_not_block_next_poll(adapter):
    calls = []

    def call_async(request):
        calls.append(request)
        if len(calls) == 1:
            raise RuntimeError('service disappeared during discovery')
        return Future(SimpleNamespace(current_state=SimpleNamespace(id=3)))

    adapter._lifecycle_client = SimpleNamespace(service_is_ready=lambda: True, call_async=call_async)

    adapter._check_lifecycle()
    assert not adapter.ready()
    assert adapter._lifecycle_future is None
    adapter._check_lifecycle()
    assert adapter.ready()
    assert len(calls) == 2


def goto_harness(adapter, nav2_module, monkeypatch, *, pose_age=0.0, pose=None, handle=None):
    """One control cycle, then operator preemption; no real ROS executor."""
    published, ticks = [], []
    costmap = small_costmap()
    handle = handle or GoalHandle()
    adapter.client.send_goal_async = lambda *args, **kwargs: Future(handle)
    adapter.publish_map = lambda: None
    adapter.node = SimpleNamespace(
        costmap=costmap, scan=full_scan(stamp=10.7),
        world_pose=SimpleNamespace(header=velocity_message(stamp=10.7 - pose_age).header),
        wait_for_sensors=lambda: True, preempted=lambda: len(ticks) >= 2,
        publish=published.append, pose=lambda: pose or Pose(-0.5, 0.5, 0.0),
        now=lambda: 10.7 if ticks else 10.0,
        get_clock=lambda: SimpleNamespace(now=lambda: SimpleNamespace(
            to_msg=lambda: nav2_module.TransformStamped().header.stamp)),
    )
    monkeypatch.setattr(nav2_module.rclpy, 'ok', lambda: True)
    monkeypatch.setattr(nav2_module.rclpy, 'spin_once', lambda *args, **kwargs: ticks.append(True))
    return published, ticks, costmap.cell_to_world(2, 2)


@pytest.mark.parametrize(('age', 'expected'), [(0.2, Command(0.1)), (0.6, Command())])
def test_goto_only_forwards_recent_velocity(adapter, nav2_module, monkeypatch, age, expected):
    published, ticks, goal = goto_harness(adapter, nav2_module, monkeypatch)

    def spin_once(*args, **kwargs):
        ticks.append(True)
        if len(ticks) == 1:
            adapter._on_velocity(velocity_message(stamp=10.7 - age))

    monkeypatch.setattr(nav2_module.rclpy, 'spin_once', spin_once)

    result = adapter.goto(*goal, 20.0)

    assert result['reason'] == 'preempted'
    assert published[0] == published[-1] == Command()
    assert published[1] == expected
    assert all(command == Command() for command in published[2:])
    assert not adapter._accepted
    assert adapter._velocity is None


@pytest.mark.parametrize('pose_age', [0.8, -0.2])
def test_fresh_lidar_cannot_authorize_motion_with_stale_or_future_pose(
    adapter, nav2_module, monkeypatch, pose_age,
):
    published, _, goal = goto_harness(adapter, nav2_module, monkeypatch, pose_age=pose_age)

    result = adapter.goto(*goal, 20.0)

    assert result['status'] == FAILED
    assert 'stale physical world pose' in result['reason']
    assert published and all(command == Command() for command in published)


@pytest.mark.parametrize(('pose', 'status'), [
    (Pose(-0.5, 0.5, 0.0), DONE), (Pose(-1.0, 0.5, 0.0), FAILED),
])
def test_nav2_success_requires_physical_pose_at_goal(adapter, nav2_module, monkeypatch, pose, status):
    handle = GoalHandle()
    handle.result_future = Future(SimpleNamespace(
        status=nav2_module.GoalStatus.STATUS_SUCCEEDED, result=SimpleNamespace()))
    published, _, goal = goto_harness(adapter, nav2_module, monkeypatch, pose=pose, handle=handle)

    result = adapter.goto(*goal, 20.0)

    assert result['status'] == status
    if status == FAILED:
        assert 'outside goal tolerance' in result['reason']
    assert published and all(command == Command() for command in published)
