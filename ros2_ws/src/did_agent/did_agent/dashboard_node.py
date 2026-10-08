"""ROS 2 node serving the web dashboard (http://localhost:8080)."""

from __future__ import annotations

from math import atan2
from pathlib import Path
import json

from nav_msgs.msg import Odometry
import rclpy
from rclpy.node import Node
from std_msgs.msg import String

from did_agent.dashboard_core import DashboardData
from did_agent.dashboard_core import DashboardServer
from did_agent.dashboard_core import render_geometry
from did_agent.dashboard_core import truth_from_scenario
from did_agent.grid import load_map

try:
    from did_judge.scenario import load_scenario
    from did_judge.scenario import scenario_path
except ImportError:  # the dashboard also works without the judge package
    load_scenario = scenario_path = None


class DashboardNode(Node):
    """Collect what the agent publishes and show it on a web page."""

    def __init__(self) -> None:
        super().__init__('dashboard')
        self.declare_parameter('port', 8080)
        self.declare_parameter('base_x', -2.0)
        self.declare_parameter('base_y', -0.5)
        base = (float(self.get_parameter('base_x').value),
                float(self.get_parameter('base_y').value))
        self.data = DashboardData(base)
        self._truth_for: str | None = None

        geometry = render_geometry(load_map())
        self._plan_pub = self.create_publisher(String, '/agent/plan', 10)
        self._command_pub = self.create_publisher(String, '/agent/command', 10)
        self.server = DashboardServer(
            self.data, geometry, self._send_plan, self._send_command,
            port=int(self.get_parameter('port').value),
            log=self.get_logger().info,
        )
        self.server.start()

        self.create_subscription(Odometry, '/odom', self._on_odom, 10)
        self._json_topic('/agent/state', self.data.on_state)
        self._json_topic('/agent/status', self.data.on_status)
        self._json_topic('/agent/costmap', self.data.on_costmap)
        self._json_topic('/agent/journal', self.data.on_journal)
        self._json_topic('/did/events', self.data.on_event, depth=50)
        self._json_topic('/did/score', self._on_score)
        self.create_subscription(String, '/agent/plan', self._on_plan, 10)
        self.get_logger().info(f'dashboard on http://localhost:{self.server.port}')

    def _json_topic(self, topic: str, handler, depth: int = 10) -> None:
        def callback(message: String) -> None:
            try:
                handler(json.loads(message.data))
            except ValueError:
                pass
        self.create_subscription(String, topic, callback, depth)

    def _on_odom(self, message: Odometry) -> None:
        position = message.pose.pose.position
        q = message.pose.pose.orientation
        yaw = atan2(2.0 * (q.w * q.z + q.x * q.y), 1.0 - 2.0 * (q.y * q.y + q.z * q.z))
        self.data.on_pose(self.data.base[0] + position.x, self.data.base[1] + position.y, yaw)

    def _on_score(self, score: dict) -> None:
        self.data.on_score(score)
        name = score.get('scenario')
        if name and name != self._truth_for and scenario_path is not None:
            self._truth_for = name
            try:
                path = score.get('scenario_file')
                if not path or not Path(path).exists():
                    path = scenario_path(name)
                self.data.truth = truth_from_scenario(load_scenario(path))
            except (OSError, ValueError, KeyError) as error:
                self.get_logger().warning(f'no ground truth for {name!r}: {error}')

    def _on_plan(self, message: String) -> None:
        self.data.on_plan(message.data)

    def _send_plan(self, text: str) -> None:
        self._plan_pub.publish(String(data=text))

    def _send_command(self, command: str) -> None:
        self._command_pub.publish(String(data=json.dumps({'cmd': command})))


def main(args=None) -> None:
    """``ros2 run did_agent dashboard``."""
    rclpy.init(args=args)
    node = DashboardNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.server.stop()
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
