"""ROS 2 node exposing the task judge interface (/did/*)."""

from __future__ import annotations

import json
from pathlib import Path
import re
from typing import Any

from nav_msgs.msg import Odometry
import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import LaserScan
from std_msgs.msg import Float32
from std_msgs.msg import String
from std_srvs.srv import Trigger

from did_judge.judge_model import JudgeModel
from did_judge.judge_model import scan_clearance
from did_judge.judge_model import score_payload
from did_judge.scenario import load_scenario
from did_judge.scenario import scenario_path

SCENARIOS = ('easy', 'medium', 'hard')
SEEDED_SCENARIO = re.compile(r'^(easy|medium|hard)@(0|[1-9][0-9]{0,9})$')
GENERATED_SCENARIO_DIR = Path('/tmp/scenarios')


def scenario_request(text: str) -> tuple[str, Path]:
    """Resolve a trusted built-in or dashboard-generated scenario request."""
    raw = text.strip()
    try:
        payload = json.loads(raw)
    except (TypeError, ValueError):
        payload = raw
    if isinstance(payload, dict):
        name = payload.get('name')
        supplied_path = payload.get('scenario_file')
    else:
        name = payload
        supplied_path = None
    if not isinstance(name, str):
        raise ValueError('scenario name must be a string')
    name = name.strip().lower()
    if name in SCENARIOS:
        return name, scenario_path(name)
    match = SEEDED_SCENARIO.fullmatch(name)
    if match is None or int(match.group(2)) > 2_147_483_647:
        raise ValueError(f'unknown scenario {name!r}')
    if not isinstance(supplied_path, str):
        raise ValueError('generated scenario requires scenario_file')
    path = Path(supplied_path).resolve()
    root = GENERATED_SCENARIO_DIR.resolve()
    if path.parent != root or path.suffix != '.yaml':
        raise ValueError('generated scenario_file is outside the allowed directory')
    return name, path


class JudgeNode(Node):
    """Publish battery, sensor, score and penalty events; serve collect/finish."""

    def __init__(self) -> None:
        super().__init__('judge')
        self.declare_parameter('scenario_file', '')
        self.declare_parameter('collision_clearance', 0.015)
        self.declare_parameter('collision_cooldown', 2.0)

        scenario_file = str(self.get_parameter('scenario_file').value)
        if not scenario_file:
            raise RuntimeError('parameter scenario_file is required')
        scenario = load_scenario(scenario_file)
        self._collision_clearance = float(self.get_parameter('collision_clearance').value)
        self._collision_cooldown = float(self.get_parameter('collision_cooldown').value)
        self._scenario_file = scenario_file
        self._model = JudgeModel(
            scenario,
            collision_clearance=self._collision_clearance,
            collision_cooldown=self._collision_cooldown,
        )
        self._start_time: float | None = None
        self._logged_environment = 0

        self._battery_publisher = self.create_publisher(Float32, '/did/battery', 10)
        self._sensor_publisher = self.create_publisher(
            Float32,
            '/did/sample_sensor',
            10,
        )
        self._score_publisher = self.create_publisher(String, '/did/score', 10)
        self._events_publisher = self.create_publisher(String, '/did/events', 10)
        self.create_subscription(Odometry, '/odom', self._on_odometry, 10)
        self.create_subscription(String, '/did/scenario/select', self._on_scenario_select, 10)
        self.create_subscription(
            LaserScan,
            '/scan',
            self._on_scan,
            qos_profile_sensor_data,
        )
        self.create_service(Trigger, '/did/collect', self._on_collect)
        self.create_service(Trigger, '/did/finish', self._on_finish)
        self.create_timer(0.10, self._publish_state)

        self.get_logger().info(
            f'Judge ready: scenario={scenario.name}, '
            f'samples={len(self._model.samples)}, battery={self._model.battery:.1f}'
        )

    def _on_scenario_select(self, message: String) -> None:
        """Start a fresh built-in or reproducibly generated episode."""
        try:
            name, path = scenario_request(message.data)
            scenario = load_scenario(path)
            if scenario.name != name:
                raise ValueError(
                    f'scenario file contains {scenario.name!r}, expected {name!r}'
                )
        except (OSError, ValueError, KeyError) as error:
            self.get_logger().error(f'could not load scenario request: {error}')
            return
        self._scenario_file = str(path)
        self._model = JudgeModel(
            scenario,
            collision_clearance=self._collision_clearance,
            collision_cooldown=self._collision_cooldown,
        )
        self._start_time = None
        self._logged_environment = 0
        self._publish_event('scenario_selected', scenario=name)
        self.get_logger().info(
            f'Judge reset: scenario={scenario.name}, '
            f'samples={len(self._model.samples)}, battery={self._model.battery:.1f}'
        )

    def _now(self) -> float:
        """Return seconds since the first odometry message (simulation time)."""
        now = self.get_clock().now().nanoseconds * 1e-9
        if self._start_time is None:
            return 0.0
        return now - self._start_time

    def _tick(self) -> None:
        self._model.advance(self._now())
        for entry in self._model.environment_log[self._logged_environment:]:
            # Only the judge's own log: the agent must notice changes itself.
            self.get_logger().info(f'environment change applied: {entry}')
        self._logged_environment = len(self._model.environment_log)

    def _on_odometry(self, message: Odometry) -> None:
        if self._start_time is None:
            self._start_time = self.get_clock().now().nanoseconds * 1e-9
        self._tick()
        position = message.pose.pose.position
        for event in self._model.update_odometry(position.x, position.y):
            self._publish_event(**event)

    def _on_scan(self, message: LaserScan) -> None:
        angles = [
            message.angle_min + i * message.angle_increment
            for i in range(len(message.ranges))
        ]
        clearance = scan_clearance(angles, message.ranges, message.range_min)
        if clearance is None:
            return
        self._tick()
        event = self._model.report_scan(clearance)
        if event is not None:
            self._publish_event(**event)

    def _publish_state(self) -> None:
        self._tick()
        battery = Float32()
        battery.data = float(self._model.battery)
        self._battery_publisher.publish(battery)

        sensor = Float32()
        sensor.data = float(self._model.sample_sensor())
        self._sensor_publisher.publish(sensor)

        model = self._model
        score = String()
        score.data = json.dumps(
            score_payload(model, self._scenario_file),
            sort_keys=True,
        )
        self._score_publisher.publish(score)

    def _publish_event(self, event: str, **details: Any) -> None:
        message = String()
        message.data = json.dumps(
            {'event': event, 't': round(self._model.time, 2), **details},
            sort_keys=True,
        )
        self._events_publisher.publish(message)

    def _on_collect(self, _request: Trigger.Request, response: Trigger.Response):
        success, event = self._model.collect()
        response.success = success
        response.message = (
            'sample collected' if success else f'no sample within '
            f'{self._model.collection_radius:.2f} m'
        )
        self._publish_event(**event)
        return response

    def _on_finish(self, _request: Trigger.Request, response: Trigger.Response):
        response.success = self._model.finish()
        response.message = (
            'run finished at base' if response.success else 'robot is not at base'
        )
        return response


def main(args=None) -> None:
    """Run the judge node."""
    rclpy.init(args=args)
    node = JudgeNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
