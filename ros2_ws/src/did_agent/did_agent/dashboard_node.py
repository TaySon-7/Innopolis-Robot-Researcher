"""ROS 2 node serving the web dashboard (http://localhost:8080)."""

from __future__ import annotations

from ament_index_python.packages import get_package_share_directory
from math import atan2
from pathlib import Path
from threading import Event
from time import monotonic
from time import sleep
import json

from nav_msgs.msg import Odometry
import rclpy
from rclpy.node import Node
from ros_gz_interfaces.srv import DeleteEntity
from ros_gz_interfaces.srv import SpawnEntity
from std_msgs.msg import String

from did_agent.dashboard_core import DashboardData
from did_agent.dashboard_core import DashboardServer
from did_agent.dashboard_core import preview_from_scenario
from did_agent.dashboard_core import render_geometry
from did_agent.dashboard_core import truth_from_scenario
from did_agent.gazebo_geometry import request_gazebo_geometry
from did_agent.grid import load_map
from did_agent.scenario_generator import load_named
from did_agent.scenario_generator import to_yaml

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
        self._truth_scenario = None
        self._scenario_cache: dict[str, object] = {}

        self._fallback_geometry = render_geometry(load_map())
        geometry = request_gazebo_geometry(
            self._fallback_geometry,
            timeout=800,
            log=self.get_logger().warning,
        )
        self._plan_pub = self.create_publisher(String, '/agent/plan', 10)
        self._command_pub = self.create_publisher(String, '/agent/command', 10)
        self._scenario_pub = self.create_publisher(String, '/did/scenario/select', 10)
        self._delete_entity = self.create_client(
            DeleteEntity,
            '/world/default/remove/blocking',
        )
        self._spawn_entity = self.create_client(
            SpawnEntity,
            '/world/default/create/blocking',
        )
        self._burger_sdf = (
            Path(get_package_share_directory('turtlebot3_gazebo'))
            / 'models' / 'turtlebot3_burger' / 'model.sdf'
        )
        self.server = DashboardServer(
            self.data, geometry, self._send_plan, self._send_command,
            self._send_scenario, self._preview_scenario,
            port=int(self.get_parameter('port').value),
            log=self.get_logger().info,
        )
        self.server.start()
        self._geometry_timer = None
        if geometry.get('source') != 'gazebo_scene':
            self._geometry_timer = self.create_timer(2.0, self._refresh_geometry)

        self.create_subscription(Odometry, '/odom', self._on_odom, 10)
        self._json_topic('/agent/state', self.data.on_state)
        self._json_topic('/agent/status', self.data.on_status)
        self._json_topic('/agent/costmap', self.data.on_costmap)
        self._json_topic('/agent/journal', self.data.on_journal)
        self._json_topic('/agent/hypotheses', self.data.on_hypotheses)
        self._json_topic('/did/events', self.data.on_event, depth=50)
        self._json_topic('/did/score', self._on_score)
        self.create_subscription(String, '/agent/plan', self._on_plan, 10)
        self.get_logger().info(f'dashboard on http://localhost:{self.server.port}')

    def _refresh_geometry(self) -> None:
        """Retry until Gazebo's scene broadcaster is ready during startup."""
        geometry = request_gazebo_geometry(self._fallback_geometry, timeout=800)
        if geometry.get('source') != 'gazebo_scene':
            return
        self.server.geometry = geometry
        if self._geometry_timer is not None:
            self._geometry_timer.cancel()
        self.get_logger().info('dashboard geometry loaded from Gazebo scene/info')

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
                self._truth_scenario = load_scenario(path)
            except (OSError, ValueError, KeyError) as error:
                self._truth_scenario = None
                self.get_logger().warning(f'no ground truth for {name!r}: {error}')
        if self._truth_scenario is not None:
            try:
                at = float(score.get('t', 0.0))
            except (TypeError, ValueError):
                at = 0.0
            self.data.on_truth(truth_from_scenario(self._truth_scenario, at))

    def _on_plan(self, message: String) -> None:
        self.data.on_plan(message.data)

    def _send_plan(self, text: str) -> None:
        self._plan_pub.publish(String(data=text))

    def _send_command(self, command: str) -> None:
        self._command_pub.publish(String(data=json.dumps({'cmd': command})))

    def _load_named_scenario(self, name: str):
        """Load once so the preview and launched episode share the exact object."""
        scenario = self._scenario_cache.get(name)
        if scenario is None:
            scenario = load_named(name)
            if len(self._scenario_cache) >= 24:
                self._scenario_cache.pop(next(iter(self._scenario_cache)))
            self._scenario_cache[name] = scenario
        return scenario

    def _preview_scenario(self, name: str) -> dict:
        return preview_from_scenario(self._load_named_scenario(name))

    def _send_scenario(self, scenario: str) -> None:
        """Reset Gazebo, then ask the judge and agent to load a clean scenario."""
        selected = self._load_named_scenario(scenario)
        if '@' in scenario:
            generated_dir = Path('/tmp/scenarios')
            generated_dir.mkdir(parents=True, exist_ok=True)
            path = generated_dir / f"ui-{scenario.replace('@', '-')}.yaml"
            path.write_text(to_yaml(selected), encoding='utf-8')
        else:
            path = scenario_path(scenario)
        self._send_command('stop')
        self._wait_for_agent_stop()
        self._reset_gazebo()
        self.data.reset_run(reset_pose=True)
        self._truth_for = selected.name
        self._truth_scenario = selected
        self.data.on_truth(truth_from_scenario(selected, 0.0))
        request = json.dumps({
            'name': selected.name,
            'scenario_file': str(path),
        }, sort_keys=True)
        self._scenario_pub.publish(String(data=request))
        self.get_logger().info(f'scenario selection requested: {scenario}')

    def _wait_for_agent_stop(self, timeout: float = 3.0) -> None:
        """Do not respawn Burger while an old plan can still call ``finish``."""
        deadline = monotonic() + timeout
        while self.data.run_state() == 'running':
            if monotonic() >= deadline:
                raise RuntimeError('Agent did not stop before Gazebo restart')
            sleep(0.02)

    def _reset_gazebo(self, timeout: float = 4.0) -> None:
        """Recreate Burger without corrupting its Gazebo plugins.

        Gazebo Harmonic's ``reset.all`` invalidates link and joint entities for
        a model spawned after world load.  Removing and spawning Burger gives
        us fresh diff-drive, odom and lidar systems while preserving the static
        arena.  Episode time is reset by the judge together with the scenario;
        resetting the global Gazebo clock races dynamic model creation.
        """
        deadline = monotonic() + timeout
        clients = (
            (self._delete_entity, 'remove'),
            (self._spawn_entity, 'create'),
        )
        for client, name in clients:
            remaining = max(0.0, deadline - monotonic())
            if not client.wait_for_service(timeout_sec=remaining):
                raise RuntimeError(f'Gazebo {name} service is unavailable')

        remove = DeleteEntity.Request()
        remove.entity.name = 'burger'
        remove.entity.type = remove.entity.MODEL
        self._call_gazebo(self._delete_entity, remove, 'remove Burger', deadline)

        spawn = SpawnEntity.Request()
        spawn.entity_factory.name = 'burger'
        spawn.entity_factory.allow_renaming = False
        spawn.entity_factory.sdf_filename = str(self._burger_sdf)
        spawn.entity_factory.pose.position.x = self.data.base[0]
        spawn.entity_factory.pose.position.y = self.data.base[1]
        spawn.entity_factory.pose.position.z = 0.01
        spawn.entity_factory.pose.orientation.w = 1.0
        spawn.entity_factory.relative_to = 'world'

        try:
            # Even the blocking remove service returns before every system has
            # observed the entity removal.  One short server-side barrier keeps
            # a delayed removal command from deleting the replacement model.
            sleep(0.25)
        finally:
            self._call_gazebo(
                self._spawn_entity,
                spawn,
                'spawn Burger',
                monotonic() + timeout,
            )
        self.get_logger().info('Gazebo restarted: Burger respawned at the base')

    @staticmethod
    def _call_gazebo(client, request, action: str, deadline: float) -> None:
        """Wait for one Gazebo service response while the ROS executor spins."""
        future = client.call_async(request)
        completed = Event()
        future.add_done_callback(lambda _future: completed.set())
        if not completed.wait(max(0.0, deadline - monotonic())):
            raise RuntimeError(f'Gazebo did not confirm {action} in time')
        try:
            response = future.result()
        except Exception as error:
            raise RuntimeError(f'Gazebo {action} failed: {error}') from error
        if response is None or not response.success:
            raise RuntimeError(f'Gazebo rejected {action}')


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
