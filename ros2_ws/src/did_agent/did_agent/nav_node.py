"""ROS 2 wrapper around the navigator and the ``goto`` command-line tool."""

from __future__ import annotations

import argparse
from math import atan2
from math import hypot
import json
import sys
import time

import numpy as np
from geometry_msgs.msg import PoseStamped
from geometry_msgs.msg import TwistStamped
import rclpy
from rclpy.node import Node
from rclpy.parameter import Parameter
from rclpy.qos import qos_profile_sensor_data
from rclpy.utilities import remove_ros_args
from sensor_msgs.msg import LaserScan

from did_agent.controller import Command
from did_agent.controller import Pose
from did_agent.costmap import CostMap
from did_agent.navigator_core import DONE
from did_agent.navigator_core import FAILED
from did_agent.navigator_core import NavigatorCore
from did_agent.navigator_core import Scan

BASE_X = -2.0
BASE_Y = -0.5
LOOP_PERIOD = 0.05
# hls_lfcd_lds sensor pose in TurtleBot3 Burger's Gazebo model.sdf.
LIDAR_X_OFFSET = -0.032
WORLD_POSE_TIMEOUT = 0.75
WORLD_POSE_FUTURE_TOLERANCE = 0.1


def world_pose_fresh(
    message: PoseStamped | None,
    now: float,
    timeout: float = WORLD_POSE_TIMEOUT,
) -> bool:
    """Return whether ``message`` is recent enough for safe motion."""
    if message is None:
        return False
    stamp = message.header.stamp.sec + message.header.stamp.nanosec * 1e-9
    age = now - stamp
    return -WORLD_POSE_FUTURE_TOLERANCE <= age <= timeout


def pose_from_world_message(message: PoseStamped) -> Pose:
    """Convert the normalized physical world pose to the navigation type."""
    position = message.pose.position
    q = message.pose.orientation
    yaw = atan2(2.0 * (q.w * q.z + q.x * q.y), 1.0 - 2.0 * (q.y * q.y + q.z * q.z))
    return Pose(position.x, position.y, yaw)


class Navigator(Node):
    """Own the sensors and /cmd_vel, and drive to goals with NavigatorCore."""

    def __init__(self, name: str = 'navigator') -> None:
        super().__init__(
            name,
            parameter_overrides=[Parameter('use_sim_time', value=True)],
        )
        self.declare_parameter('base_x', BASE_X)
        self.declare_parameter('base_y', BASE_Y)
        self.declare_parameter('navigation_backend', 'custom')
        self.navigation_backend = self.get_parameter('navigation_backend').value
        if self.navigation_backend not in ('custom', 'nav2'):
            raise ValueError('navigation_backend must be custom or nav2')
        self.base_x = float(self.get_parameter('base_x').value)
        self.base_y = float(self.get_parameter('base_y').value)
        self.costmap = CostMap()
        self.core = NavigatorCore(self.costmap)
        self.world_pose: PoseStamped | None = None
        self.scan: Scan | None = None
        self._velocity = self.create_publisher(TwistStamped, '/cmd_vel', 10)
        self.create_subscription(PoseStamped, '/did/world_pose', self._on_world_pose, 10)
        self.create_subscription(
            LaserScan,
            '/scan',
            self._on_scan,
            qos_profile_sensor_data,
        )
        # Imported here so pure navigation modules remain usable without ROS
        # Nav2 packages on a prepared host.
        from did_agent.nav2_adapter import Nav2Adapter
        self._nav2 = Nav2Adapter(self)

    def _on_world_pose(self, message: PoseStamped) -> bool:
        if not world_pose_fresh(message, self.now()):
            return False
        self.world_pose = message
        if getattr(self, '_nav2', None) is not None:
            self._nav2.on_pose(message)
        return True

    def clear_world_pose(self) -> None:
        """Discard a pose belonging to the previous Burger instance."""
        self.world_pose = None
        self.scan = None

    def _on_scan(self, message: LaserScan) -> None:
        self.scan = Scan(
            angle_min=message.angle_min,
            angle_increment=message.angle_increment,
            ranges=np.asarray(message.ranges, dtype=float),
            range_min=message.range_min,
            stamp=message.header.stamp.sec + message.header.stamp.nanosec * 1e-9,
            x_offset=LIDAR_X_OFFSET,
        )
        if getattr(self, '_nav2', None) is not None:
            self._nav2.on_scan(message)

    def now(self) -> float:
        """Return simulation time in seconds."""
        return self.get_clock().now().nanoseconds * 1e-9

    def pose(self) -> Pose | None:
        """Return the last known physical pose for read-only calculations."""
        if self.world_pose is None:
            return None
        return pose_from_world_message(self.world_pose)

    def fresh_pose(self) -> Pose | None:
        """Return a physical pose only while it is fresh enough for motion."""
        if not world_pose_fresh(self.world_pose, self.now()):
            return None
        return pose_from_world_message(self.world_pose)

    def publish(self, command: Command) -> None:
        """Publish a velocity command."""
        message = TwistStamped()
        message.header.stamp = self.get_clock().now().to_msg()
        message.twist.linear.x = float(command.linear)
        message.twist.angular.z = float(command.angular)
        self._velocity.publish(message)

    def stop(self) -> None:
        """Command zero velocity."""
        if getattr(self, '_nav2', None) is not None:
            self._nav2.cancel()
        self.publish(Command())

    def navigation_state(self) -> dict[str, object]:
        """Common dashboard contract for the selected navigation implementation."""
        if self.navigation_backend == 'nav2':
            state = self._nav2.snapshot()
        else:
            state = {'status': self.core.status, 'replans': self.core.replans,
                     'waypoints': self.core.waypoints[:80] if self.core.status == 'running' else [],
                     'ready': self.ready(), 'reason': self.core.reason}
        state.update(backend=self.navigation_backend, available_backends=['custom', 'nav2'])
        return state

    def preempted(self) -> bool:
        """Return True when the current drive must be abandoned (override)."""
        return False

    def ready(self) -> bool:
        """Return whether all the inputs this node needs have arrived."""
        return world_pose_fresh(self.world_pose, self.now()) and self.scan is not None

    def wait_for_sensors(self, timeout: float = 60.0) -> bool:
        """Spin until the physical world pose and a scan have arrived."""
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            if self.ready():
                return True
            rclpy.spin_once(self, timeout_sec=0.1)
        return False

    def goto(
        self,
        x: float,
        y: float,
        timeout: float = 180.0,
        guard=None,
    ) -> dict[str, object]:
        """Drive to a world point; block until done, failed or timed out."""
        if getattr(self, 'navigation_backend', 'custom') == 'nav2':
            return self._nav2.goto(x, y, timeout, guard)
        if not self.ready():
            self.stop()
        if not self.wait_for_sensors():
            self.stop()
            return {'status': FAILED, 'reason': 'no physical world pose or scan'}
        pose = self.fresh_pose()
        if pose is None:
            self.stop()
            return {'status': FAILED, 'reason': 'physical world pose unavailable'}
        if not self.core.set_goal((x, y), pose, self.now()):
            self.stop()
            return self._result(pose)
        started = self.now()
        next_guard = started + 1.0
        while rclpy.ok():
            rclpy.spin_once(self, timeout_sec=LOOP_PERIOD)
            # A reset/stop may run in spin_once. Never publish another movement
            # before honoring it, and retain the last valid pose for the result.
            if self.preempted():
                self.core.cancel()
                self.stop()
                return {**self._result(pose), 'status': FAILED, 'reason': 'preempted'}
            current_pose = self.fresh_pose()
            if current_pose is None:
                self.core.cancel()
                self.stop()
                return {**self._result(pose), 'status': FAILED,
                        'reason': 'physical world pose unavailable'}
            pose = current_pose
            command = self.core.update(pose, self.scan, self.now())
            self.publish(command)
            if self.core.status in (DONE, FAILED):
                break
            if guard is not None and self.now() >= next_guard:
                next_guard = self.now() + 1.0
                if guard():
                    self.core.cancel()
                    self.stop()
                    return {**self._result(pose), 'status': FAILED,
                            'reason': 'battery reserve reached'}
            if self.now() - started > timeout:
                self.core.cancel()
                self.stop()
                return {**self._result(pose), 'status': FAILED, 'reason': 'timeout'}
        self.stop()
        return self._result(pose)

    def _result(self, pose: Pose) -> dict[str, object]:
        return {
            'status': self.core.status,
            'reason': self.core.reason,
            'pose': {'x': round(pose.x, 3), 'y': round(pose.y, 3)},
            'distance_to_goal': round(self.core.distance_to_goal(pose), 3),
            'replans': self.core.replans,
        }


def main(args=None) -> None:
    """Send goto through the agent, preserving its selected navigation backend."""
    from did_agent.navigation_cli import main as client_main

    client_main(args)


if __name__ == '__main__':
    main()
