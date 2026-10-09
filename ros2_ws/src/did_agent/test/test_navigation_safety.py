"""Regressions for contact, blind reversing and an indefinitely stuck drive."""

from math import pi

import numpy as np
import pytest

from did_agent.controller import Command, Pose
from did_agent.costmap import CostMap
from did_agent.kinematic_sim import KinematicSim
from did_agent.navigator_core import DONE, FAILED, NavigatorCore, Scan


@pytest.fixture
def costmap():
    return CostMap()


def scan_with_hits(*hits, stamp=None):
    ranges = np.full(360, np.inf)
    for angle, distance in hits:
        ranges[round(angle * 180 / pi) % 360] = distance
    return Scan(0.0, pi / 180, ranges, stamp=stamp)


@pytest.mark.parametrize('angle', [0.0, 1.5, -1.5])
def test_minimum_range_return_is_an_obstacle(costmap, angle):
    nav = NavigatorCore(costmap)
    pose = Pose(-2.0, -0.5, 0.0)
    assert nav.set_goal((-0.55, -0.55), pose, 0.0)

    command = nav.update(pose, scan_with_hits((angle, 0.12)), 0.0)

    assert command.linear == 0.0


def test_known_wall_stops_motion_before_contact(costmap, monkeypatch):
    # A static-map explanation must never disable the lidar safety clearance.
    monkeypatch.setattr(costmap, 'near_known_solid', lambda points: np.ones(len(points), bool))
    pose = Pose(-2.0, -0.5, 0.0)
    nav = NavigatorCore(costmap)
    assert nav.set_goal((-0.55, -0.55), pose, 0.0)

    assert nav.update(pose, scan_with_hits((0.0, 0.19)), 0.0).linear == 0.0


def test_backoff_never_reverses_into_a_rear_obstacle(costmap):
    pose = Pose(-2.0, -0.5, 0.0)
    nav = NavigatorCore(costmap)
    assert nav.set_goal((-0.55, -0.55), pose, 0.0)
    scan = scan_with_hits((0.0, 0.19), (pi, 0.12))
    assert nav.update(pose, scan, 0.0).linear == 0.0

    command = nav.update(pose, scan, 0.1)

    assert command == Command()
    assert nav.status == FAILED
    assert 'behind' in nav.reason


def test_backoff_checks_new_rear_obstacles_on_every_update(costmap):
    pose = Pose(-2.0, -0.5, 0.0)
    nav = NavigatorCore(costmap)
    assert nav.set_goal((-0.55, -0.55), pose, 0.0)
    nav.update(pose, scan_with_hits((0.0, 0.19)), 0.0)
    assert nav.update(pose, scan_with_hits((0.0, 0.19)), 0.1).linear < 0.0

    assert nav.update(pose, scan_with_hits((0.0, 0.19), (pi, 0.12)), 0.2) == Command()


def test_backoff_requires_rear_lidar_coverage(costmap):
    pose = Pose(-2.0, -0.5, 0.0)
    nav = NavigatorCore(costmap)
    assert nav.set_goal((-0.55, -0.55), pose, 0.0)
    front_only = Scan(0.0, 0.1, np.array([0.19]))
    nav.update(pose, front_only, 0.0)

    assert nav.update(pose, front_only, 0.1) == Command()
    assert nav.status == FAILED


@pytest.mark.parametrize('scan', [None, Scan(0.0, pi / 180, np.full(360, np.nan)),
                                 scan_with_hits(stamp=0.0)])
def test_missing_invalid_or_stale_scan_stops_the_robot(costmap, scan):
    pose = Pose(-2.0, -0.5, 0.0)
    nav = NavigatorCore(costmap)
    assert nav.set_goal((-0.55, -0.55), pose, 1.0)

    assert nav.update(pose, scan, 1.0) == Command()
    assert nav.status == FAILED
    assert 'scan' in nav.reason


def test_turning_in_place_checks_rear_clearance(costmap):
    pose = Pose(-2.0, -0.5, pi)
    nav = NavigatorCore(costmap)
    assert nav.set_goal((-0.55, -0.55), pose, 0.0)

    assert nav.update(pose, scan_with_hits((pi, 0.12)), 0.0) == Command()


@pytest.mark.parametrize('distance', [float('nan'), -float('inf'), 0.0, 0.11])
def test_invalid_forward_beam_cannot_be_treated_as_clear_space(costmap, distance):
    pose = Pose(-2.0, -0.5, 0.0)
    nav = NavigatorCore(costmap)
    assert nav.set_goal((-0.55, -0.55), pose, 0.0)

    assert nav.update(pose, scan_with_hits((0.0, distance)), 0.0) == Command()
    assert nav.status == FAILED


def test_cost_updates_cannot_reset_the_stuck_watchdog(costmap):
    pose = Pose(-2.0, -0.5, 0.0)
    nav = NavigatorCore(costmap, stuck_window=2.0, max_replans=1)
    assert nav.set_goal((-0.55, -0.55), pose, 0.0)
    scan = scan_with_hits()

    for t in np.arange(0.0, 12.0, 0.25):
        costmap.update({'circle': {'x': 1.7, 'y': 0.5, 'r': 0.1}}, 1.0 + t)
        command = nav.update(pose, scan, float(t))
        if nav.status == FAILED:
            break

    assert nav.status == FAILED
    assert command == Command()
    assert 'no progress' in nav.reason


def test_a_real_detour_is_progress_even_when_moving_away_from_goal(costmap):
    nav = NavigatorCore(costmap, stuck_window=1.0)
    pose = Pose(-2.0, -0.5, 0.0)
    assert nav.set_goal((-0.55, -0.55), pose, 0.0)

    for t in np.arange(0.0, 2.5, 0.25):
        # A legitimate sideways detour increases straight-line goal distance.
        nav.update(Pose(-2.0, -0.5 + 0.15 * t, 0.0), scan_with_hits(), float(t))

    assert nav.replans == 0


@pytest.mark.parametrize(('yaw', 'expected'), [
    (0.0, (-1.842, -0.5)), (pi / 2, (-2.0, -0.342)), (pi, (-2.158, -0.5)),
])
def test_obstacles_use_the_scanner_origin_in_world_coordinates(costmap, yaw, expected):
    scan = scan_with_hits((0.0, 0.19))
    scan.x_offset = -0.032

    kind, points = NavigatorCore(costmap).obstacles_ahead(Pose(-2.0, -0.5, yaw), scan)

    assert kind is not None
    assert len(points) == 1
    assert points[0] == pytest.approx(expected)


def test_repeated_obstacle_recovery_is_bounded(costmap):
    pose = Pose(-2.0, -0.5, 0.0)
    nav = NavigatorCore(costmap, max_replans=2)
    assert nav.set_goal((-0.55, -0.55), pose, 0.0)

    for t in np.arange(0.0, 10.0, 0.1):
        command = nav.update(pose, scan_with_hits((0.0, 0.12)), float(t))
        assert command.linear <= 0.0
        if nav.status == FAILED:
            break

    assert nav.status == FAILED
    assert command == Command()


def test_hidden_obstacle_is_avoided_with_real_lidar_update_rate(costmap):
    sim = KinematicSim(costmap.grid, hidden=[(-0.55, -0.5, 0.15)], beams=360)
    nav = NavigatorCore(costmap)
    assert nav.set_goal((0.55, -0.55), sim.pose, sim.t)
    for step in range(4800):
        # Gazebo's lidar is 5 Hz; control is 20 Hz. Reuse three old frames.
        if step % 4 == 0:
            scan = sim.scan()
            scan.stamp = sim.t
        command = nav.update(sim.pose, scan, sim.t)
        sim.step(command.linear, command.angular)
        if nav.status in (DONE, FAILED):
            break

    assert nav.status == DONE, nav.reason
    assert nav.replans > 0
    assert sim.collisions == 0
