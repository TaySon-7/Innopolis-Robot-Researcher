"""ROS adapter checks for the safety data consumed by NavigatorCore."""

from types import SimpleNamespace

import numpy as np
import pytest
from sensor_msgs.msg import LaserScan

from did_agent.controller import Command, Pose
from did_agent.costmap import CostMap
from did_agent.nav_node import Navigator
from did_agent.navigator_core import FAILED, NavigatorCore


def test_ros_scan_timestamp_activates_stale_scan_stop():
    message = LaserScan()
    message.header.stamp.sec = 12
    message.header.stamp.nanosec = 250_000_000
    message.angle_min = 0.0
    message.angle_increment = float(np.pi / 180)
    message.range_min = 0.12
    message.ranges = [float('inf')] * 360
    inputs = SimpleNamespace(scan=None)

    Navigator._on_scan(inputs, message)

    assert inputs.scan.stamp == pytest.approx(12.25)
    assert inputs.scan.x_offset == pytest.approx(-0.032)
    nav = NavigatorCore(CostMap())
    pose = Pose(-2.0, -0.5, 0.0)
    assert nav.set_goal((-0.55, -0.55), pose, 12.25)
    assert nav.update(pose, inputs.scan, 12.25).linear > 0.0
    assert nav.update(pose, inputs.scan, 13.25) == Command()
    assert nav.status == FAILED
    assert nav.reason == 'stale lidar scan'


def test_robot_reset_discards_the_previous_scan_as_well_as_pose():
    inputs = SimpleNamespace(world_pose=object(), scan=object())

    Navigator.clear_world_pose(inputs)

    assert inputs.world_pose is None
    assert inputs.scan is None
