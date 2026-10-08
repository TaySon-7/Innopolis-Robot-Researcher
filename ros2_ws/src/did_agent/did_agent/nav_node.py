"""ROS 2 wrapper around the navigator and the ``goto`` command-line tool."""

from __future__ import annotations

import argparse
from math import atan2
from math import hypot
import json
import sys
import time

import numpy as np
from geometry_msgs.msg import TwistStamped
from nav_msgs.msg import Odometry
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


class Navigator(Node):
    """Own the sensors and /cmd_vel, and drive to goals with NavigatorCore."""

    def __init__(self, name: str = 'navigator') -> None:
        super().__init__(
            name,
            parameter_overrides=[Parameter('use_sim_time', value=True)],
        )
        self.declare_parameter('base_x', BASE_X)
        self.declare_parameter('base_y', BASE_Y)
        self.base_x = float(self.get_parameter('base_x').value)
        self.base_y = float(self.get_parameter('base_y').value)
        self.costmap = CostMap()
        self.core = NavigatorCore(self.costmap)
        self.odom: Odometry | None = None
        self.scan: Scan | None = None
        self._velocity = self.create_publisher(TwistStamped, '/cmd_vel', 10)
        self.create_subscription(Odometry, '/odom', self._on_odom, 10)
        self.create_subscription(
            LaserScan,
            '/scan',
            self._on_scan,
            qos_profile_sensor_data,
        )

    def _on_odom(self, message: Odometry) -> None:
        self.odom = message

    def _on_scan(self, message: LaserScan) -> None:
        self.scan = Scan(
            angle_min=message.angle_min,
            angle_increment=message.angle_increment,
            ranges=np.asarray(message.ranges, dtype=float),
            range_min=message.range_min,
        )

    def now(self) -> float:
        """Return simulation time in seconds."""
        return self.get_clock().now().nanoseconds * 1e-9

    def pose(self) -> Pose | None:
        """Return the robot pose in world coordinates (start pose + odometry)."""
        if self.odom is None:
            return None
        position = self.odom.pose.pose.position
        q = self.odom.pose.pose.orientation
        yaw = atan2(2.0 * (q.w * q.z + q.x * q.y), 1.0 - 2.0 * (q.y * q.y + q.z * q.z))
        return Pose(self.base_x + position.x, self.base_y + position.y, yaw)

    def publish(self, command: Command) -> None:
        """Publish a velocity command."""
        message = TwistStamped()
        message.header.stamp = self.get_clock().now().to_msg()
        message.twist.linear.x = float(command.linear)
        message.twist.angular.z = float(command.angular)
        self._velocity.publish(message)

    def stop(self) -> None:
        """Command zero velocity."""
        self.publish(Command())

    def preempted(self) -> bool:
        """Return True when the current drive must be abandoned (override)."""
        return False

    def ready(self) -> bool:
        """Return whether all the inputs this node needs have arrived."""
        return self.odom is not None and self.scan is not None

    def wait_for_sensors(self, timeout: float = 60.0) -> bool:
        """Spin until odometry and a scan have arrived."""
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
        if not self.wait_for_sensors():
            return {'status': FAILED, 'reason': 'no odometry or scan'}
        pose = self.pose()
        if not self.core.set_goal((x, y), pose, self.now()):
            return self._result(pose)
        started = self.now()
        next_guard = started + 1.0
        while rclpy.ok():
            rclpy.spin_once(self, timeout_sec=LOOP_PERIOD)
            pose = self.pose()
            command = self.core.update(pose, self.scan, self.now())
            self.publish(command)
            if self.core.status in (DONE, FAILED):
                break
            if self.preempted():
                self.core.cancel()
                self.stop()
                return {**self._result(pose), 'status': FAILED, 'reason': 'preempted'}
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
    """Drive the robot to a world point: ``ros2 run did_agent goto --x 1 --y 0``."""
    parser = argparse.ArgumentParser(description='Drive the robot to a world point.')
    parser.add_argument('--x', type=float, required=True)
    parser.add_argument('--y', type=float, required=True)
    parser.add_argument('--timeout', type=float, default=180.0)
    options = parser.parse_args(remove_ros_args(args if args is not None else sys.argv)[1:])

    rclpy.init(args=args)
    node = Navigator('goto')
    exit_code = 1
    try:
        result = node.goto(options.x, options.y, options.timeout)
        print(json.dumps(result, sort_keys=True))
        exit_code = 0 if result['status'] == DONE else 1
    finally:
        node.stop()
        node.destroy_node()
        rclpy.shutdown()
    raise SystemExit(exit_code)


if __name__ == '__main__':
    main()
