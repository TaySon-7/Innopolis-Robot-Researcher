"""ROS 2 node exposing the task judge interface for level zero."""

from __future__ import annotations

import json

from nav_msgs.msg import Odometry
import rclpy
from rclpy.node import Node
from std_msgs.msg import Float32
from std_msgs.msg import String
from std_srvs.srv import Trigger

from did_judge.judge_model import JudgeModel


class JudgeNode(Node):
    """Publish battery and sample sensor values and handle judge services."""

    def __init__(self) -> None:
        super().__init__('judge')
        self.declare_parameter('scenario', 'easy')
        self.declare_parameter('base_x', -2.0)
        self.declare_parameter('base_y', -0.5)
        self.declare_parameter('initial_battery', 60.0)
        self.declare_parameter('battery_cost_per_meter', 1.0)
        self.declare_parameter('collection_radius', 0.30)
        self.declare_parameter('sensor_range', 1.50)
        self.declare_parameter('sensor_noise_stddev', 0.01)
        self.declare_parameter('random_seed', 2026)
        self.declare_parameter(
            'sample_positions',
            [-1.50, -0.50, -0.75, 0.25, 0.25, -0.50],
        )

        self._scenario = str(self.get_parameter('scenario').value)
        self._model = JudgeModel(
            base_x=float(self.get_parameter('base_x').value),
            base_y=float(self.get_parameter('base_y').value),
            initial_battery=float(self.get_parameter('initial_battery').value),
            battery_cost_per_meter=float(
                self.get_parameter('battery_cost_per_meter').value
            ),
            collection_radius=float(self.get_parameter('collection_radius').value),
            sensor_range=float(self.get_parameter('sensor_range').value),
            sensor_noise_stddev=float(
                self.get_parameter('sensor_noise_stddev').value
            ),
            random_seed=int(self.get_parameter('random_seed').value),
            sample_positions=self.get_parameter('sample_positions').value,
        )

        self._battery_publisher = self.create_publisher(Float32, '/did/battery', 10)
        self._sensor_publisher = self.create_publisher(
            Float32,
            '/did/sample_sensor',
            10,
        )
        self._score_publisher = self.create_publisher(String, '/did/score', 10)
        self._events_publisher = self.create_publisher(String, '/did/events', 10)
        self.create_subscription(Odometry, '/odom', self._on_odometry, 10)
        self.create_service(Trigger, '/did/collect', self._on_collect)
        self.create_service(Trigger, '/did/finish', self._on_finish)
        self.create_timer(0.10, self._publish_state)

        self.get_logger().info(
            f'Judge ready: scenario={self._scenario}, '
            f'samples={len(self._model.samples)}, battery={self._model.battery:.1f}'
        )

    def _on_odometry(self, message: Odometry) -> None:
        position = message.pose.pose.position
        self._model.update_odometry(position.x, position.y)

    def _publish_state(self) -> None:
        battery = Float32()
        battery.data = float(self._model.battery)
        self._battery_publisher.publish(battery)

        sensor = Float32()
        sensor.data = float(self._model.sample_sensor())
        self._sensor_publisher.publish(sensor)

        score = String()
        score.data = json.dumps(
            {
                'scenario': self._scenario,
                'battery': round(self._model.battery, 3),
                'collected': self._model.collected_count,
                'samples_total': len(self._model.samples),
                'distance_travelled': round(self._model.distance_travelled, 3),
                'finished': self._model.finished,
                'world_pose': {
                    'x': round(self._model.world_x, 3),
                    'y': round(self._model.world_y, 3),
                },
            },
            sort_keys=True,
        )
        self._score_publisher.publish(score)

    def _publish_event(self, event: str, **details: object) -> None:
        message = String()
        message.data = json.dumps({'event': event, **details}, sort_keys=True)
        self._events_publisher.publish(message)

    def _on_collect(self, _request: Trigger.Request, response: Trigger.Response):
        success = self._model.collect()
        response.success = success
        if success:
            response.message = 'sample collected'
            self._publish_event(
                'sample_collected',
                collected=self._model.collected_count,
            )
        else:
            response.message = 'no sample within 0.30 m'
            self._publish_event('false_collect')
        return response

    def _on_finish(self, _request: Trigger.Request, response: Trigger.Response):
        response.success = self._model.at_base()
        if response.success:
            self._model.finished = True
            response.message = 'run finished at base'
        else:
            response.message = 'robot is not at base'
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

