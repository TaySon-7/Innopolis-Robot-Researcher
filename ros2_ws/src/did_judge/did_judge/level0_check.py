"""End-to-end smoke check for the level-zero simulation."""

from __future__ import annotations

import json
from math import hypot
import sys
import time

from geometry_msgs.msg import TwistStamped
from nav_msgs.msg import Odometry
import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import LaserScan
from std_msgs.msg import Float32


class LevelZeroCheck(Node):
    """Observe required topics and move the robot a short safe distance."""

    def __init__(self) -> None:
        super().__init__('level0_check')
        self.odom: Odometry | None = None
        self.scan: LaserScan | None = None
        self.battery: float | None = None
        self.sample_sensor: float | None = None
        self.create_subscription(Odometry, '/odom', self._on_odom, 10)
        self.create_subscription(
            LaserScan,
            '/scan',
            self._on_scan,
            qos_profile_sensor_data,
        )
        self.create_subscription(Float32, '/did/battery', self._on_battery, 10)
        self.create_subscription(
            Float32,
            '/did/sample_sensor',
            self._on_sample_sensor,
            10,
        )
        self.velocity_publisher = self.create_publisher(
            TwistStamped,
            '/cmd_vel',
            10,
        )

    def _on_odom(self, message: Odometry) -> None:
        self.odom = message

    def _on_scan(self, message: LaserScan) -> None:
        self.scan = message

    def _on_battery(self, message: Float32) -> None:
        self.battery = float(message.data)

    def _on_sample_sensor(self, message: Float32) -> None:
        self.sample_sensor = float(message.data)

    def publish_velocity(self, linear_x: float) -> None:
        message = TwistStamped()
        message.header.stamp = self.get_clock().now().to_msg()
        message.twist.linear.x = linear_x
        self.velocity_publisher.publish(message)


def _position(message: Odometry) -> tuple[float, float]:
    point = message.pose.pose.position
    return point.x, point.y


def _spin_until(node: Node, predicate, timeout: float) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        rclpy.spin_once(node, timeout_sec=0.10)
        if predicate():
            return True
    return False


def _run_check(node: LevelZeroCheck) -> dict[str, object]:
    ready = _spin_until(
        node,
        lambda: all(
            value is not None
            for value in (
                node.odom,
                node.scan,
                node.battery,
                node.sample_sensor,
            )
        ),
        timeout=90.0,
    )
    if not ready:
        missing = [
            name
            for name, value in {
                '/odom': node.odom,
                '/scan': node.scan,
                '/did/battery': node.battery,
                '/did/sample_sensor': node.sample_sensor,
            }.items()
            if value is None
        ]
        raise RuntimeError(f'timed out waiting for topics: {", ".join(missing)}')

    start_position = _position(node.odom)
    start_battery = node.battery
    drive_deadline = time.monotonic() + 20.0
    moved = 0.0
    while time.monotonic() < drive_deadline and moved < 0.08:
        node.publish_velocity(0.12)
        rclpy.spin_once(node, timeout_sec=0.05)
        moved = hypot(
            _position(node.odom)[0] - start_position[0],
            _position(node.odom)[1] - start_position[1],
        )

    for _ in range(10):
        node.publish_velocity(0.0)
        rclpy.spin_once(node, timeout_sec=0.05)

    battery_changed = _spin_until(
        node,
        lambda: node.battery < start_battery,
        timeout=5.0,
    )

    if moved < 0.05:
        raise RuntimeError(f'robot did not move far enough: {moved:.3f} m')
    if not battery_changed:
        raise RuntimeError('battery did not decrease after movement')
    if not node.scan.ranges:
        raise RuntimeError('/scan contains no lidar ranges')
    if not 0.0 <= node.sample_sensor <= 1.0:
        raise RuntimeError('/did/sample_sensor is outside [0, 1]')

    return {
        'status': 'PASS',
        'movement_m': round(moved, 3),
        'battery_before': round(start_battery, 3),
        'battery_after': round(node.battery, 3),
        'scan_samples': len(node.scan.ranges),
        'sample_sensor': round(node.sample_sensor, 3),
    }


def main(args=None) -> None:
    """Run the end-to-end check and return a shell-friendly status code."""
    rclpy.init(args=args)
    node = LevelZeroCheck()
    exit_code = 0
    try:
        result = _run_check(node)
        print(json.dumps(result, indent=2, sort_keys=True))
    except Exception as error:  # noqa: BLE001 - this is a CLI health check
        exit_code = 1
        print(f'LEVEL 0 CHECK FAILED: {error}', file=sys.stderr)
    finally:
        for _ in range(3):
            node.publish_velocity(0.0)
            rclpy.spin_once(node, timeout_sec=0.05)
        node.destroy_node()
        rclpy.shutdown()
    raise SystemExit(exit_code)


if __name__ == '__main__':
    main()

