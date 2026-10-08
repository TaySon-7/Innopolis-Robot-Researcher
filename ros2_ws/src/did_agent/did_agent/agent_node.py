"""The agent as a ROS 2 node: executes plans from /agent/plan, publishes state."""

from __future__ import annotations

import argparse
from math import hypot
from statistics import median
import json
import sys
import time
from typing import Any

from std_msgs.msg import Float32
from std_msgs.msg import String
from std_srvs.srv import Trigger
import rclpy
from rclpy.utilities import remove_ros_args

from did_agent.autonomous import AutonomousAgent
from did_agent.costmap import CostMap
from did_agent.dashboard_core import costmap_layers
from did_agent.dashboard_core import valid_scenario_name
from did_agent.executor import PlanExecutor
from did_agent.adaptation import Adaptation
from did_agent.nav_node import Navigator
from did_agent.navigator_core import NavigatorCore
from did_agent.plan import Plan
from did_agent.plan import PlanError
from did_agent.plan import parse_plan
from did_agent.robot import NavResult
from did_agent.robot import Reading
from did_agent.skills import Skills

INITIAL_BATTERY = 60.0


class AgentNode(Navigator):
    """Robot interface for skills/executor on top of the real ROS topics."""

    def __init__(self, name: str = 'agent') -> None:
        super().__init__(name)
        self._battery = INITIAL_BATTERY
        self._sensor = 0.0
        self._sensor_count = 0
        self._score: dict[str, Any] = {}
        self._pending: Plan | None = None
        self._preempt = False
        self._current: dict[str, Any] = {}
        self.cost_log: list[dict[str, Any]] = []
        self._command: str | None = None
        self._scenario_reset: str | None = None
        self._published_cost_signature: tuple[int, int] | None = None

        self.adaptation = Adaptation(
            self.costmap,
            lambda kind, title, text, status: self.journal(kind, title, text, status),
        )
        self.monitor = self.adaptation.monitor
        self.skills = Skills(self)
        self.plan_executor = PlanExecutor(self.skills, self._publish_status)

        self._status_pub = self.create_publisher(String, '/agent/status', 10)
        self._state_pub = self.create_publisher(String, '/agent/state', 10)
        self._telemetry_pub = self.create_publisher(String, '/agent/telemetry', 50)
        self._costmap_pub = self.create_publisher(String, '/agent/costmap', 5)
        self._journal_pub = self.create_publisher(String, '/agent/journal', 50)
        self.create_subscription(Float32, '/did/battery', self._on_battery, 10)
        self.create_subscription(Float32, '/did/sample_sensor', self._on_sensor, 10)
        self.create_subscription(String, '/did/score', self._on_score, 10)
        self.create_subscription(String, '/did/events', self._on_event, 50)
        self.create_subscription(String, '/agent/plan', self._on_plan, 10)
        self.create_subscription(String, '/agent/cost_update', self._on_cost_update, 10)
        self.create_subscription(String, '/agent/command', self._on_command, 10)
        self.create_subscription(String, '/did/scenario/select', self._on_scenario_select, 10)
        self._collect = self.create_client(Trigger, '/did/collect')
        self._finish = self.create_client(Trigger, '/did/finish')
        self.create_timer(1.0, self._publish_state)

    # --- callbacks ---------------------------------------------------------------------

    def _on_odom(self, message) -> None:
        super()._on_odom(message)
        pose = self.pose()
        segment = self.adaptation.on_pose(self.now(), pose.x, pose.y)
        if segment is not None:
            self._telemetry_pub.publish(String(data=json.dumps(segment)))

    def _on_battery(self, message: Float32) -> None:
        self._battery = float(message.data)
        self.adaptation.on_battery(self.now(), self._battery)

    def _on_sensor(self, message: Float32) -> None:
        self._sensor = float(message.data)
        self._sensor_count += 1
        self.adaptation.on_sensor(self.now(), self._sensor)

    def _on_score(self, message: String) -> None:
        try:
            self._score = json.loads(message.data)
        except ValueError:
            pass

    def _on_event(self, message: String) -> None:
        try:
            self.adaptation.on_event(self.now(), json.loads(message.data))
        except ValueError:
            pass

    def _on_plan(self, message: String) -> None:
        try:
            plan = parse_plan(message.data)
        except PlanError as error:
            self.get_logger().warning(f'rejected plan: {error}')
            self._publish_status({
                'plan_id': '', 'index': 0, 'type': '', 'subgoal': '',
                'state': 'failed', 'reason': f'invalid plan: {error}', 'data': {},
            })
            return
        self.get_logger().info(
            f'plan {plan.plan_id}: {[s.describe() for s in plan.subgoals]}'
        )
        self._pending = plan
        self._command = None
        self._preempt = True

    def _on_command(self, message: String) -> None:
        try:
            command = json.loads(message.data).get('cmd')
        except (ValueError, AttributeError):
            command = None
        if command not in ('auto', 'stop'):
            self.get_logger().warning(f'unknown command {message.data!r}')
            return
        self.get_logger().info(f'command: {command}')
        self._preempt = True  # interrupt whatever is running
        self._pending = None
        self._command = command if command == 'auto' else None
        if command == 'stop':
            self.core.cancel()
            self.stop()
            self.journal('decision', 'Остановка по команде оператора')

    def _on_scenario_select(self, message: String) -> None:
        """Preempt the current episode; its learned state is reset in the main loop."""
        raw = message.data.strip()
        try:
            payload = json.loads(raw)
        except ValueError:
            payload = raw
        name = payload.get('name') if isinstance(payload, dict) else payload
        if isinstance(name, str):
            name = name.strip().lower()
        if not valid_scenario_name(name):
            return
        self.get_logger().info(f'preparing clean agent state for scenario {name}')
        self._scenario_reset = name
        self._preempt = True
        self._pending = None
        self._command = None
        self.core.cancel()
        self.stop()

    def _reset_for_scenario(self, name: str) -> None:
        """Forget knowledge from the previous episode after execution has stopped."""
        self.costmap = CostMap()
        self.core = NavigatorCore(self.costmap)
        self.cost_log.clear()
        self._battery = INITIAL_BATTERY
        self._sensor = 0.0
        self._sensor_count = 0
        self._score = {}
        self._current = {}
        self._pending = None
        self._command = None
        self._preempt = False
        self._scenario_reset = None
        self._published_cost_signature = None
        self.adaptation = Adaptation(
            self.costmap,
            lambda kind, title, text, status: self.journal(kind, title, text, status),
        )
        self.monitor = self.adaptation.monitor
        self.skills = Skills(self)
        self.plan_executor = PlanExecutor(self.skills, self._publish_status)
        self._publish_status({
            'plan_id': '', 'index': 0, 'type': '', 'subgoal': '',
            'state': 'idle', 'reason': f'scenario {name} selected', 'data': {},
        })
        self._publish_costmap()
        self.journal('decision', f'Сценарий {name.upper()} загружен',
                     'Предыдущие знания сброшены. Начинаю новый независимый прогон.')

    def _on_cost_update(self, message: String) -> None:
        try:
            data = json.loads(message.data)
            region = {k: data[k] for k in ('circle', 'rect', 'cells') if k in data}
            changed = self.costmap.update(region, float(data['cost']))
            self.adaptation.on_external_update(region, float(data['cost']))
        except (ValueError, KeyError, TypeError) as error:
            self.get_logger().warning(f'rejected cost update {message.data!r}: {error}')
            return
        self.cost_log.append({'t': round(self.now(), 1), 'cost': data['cost'],
                              'source': data.get('source', ''), **region})
        self.cost_log = self.cost_log[-10:]
        self.get_logger().info(f'cost update applied to {changed} cells: {data}')

    # --- Robot interface --------------------------------------------------------------

    def ready(self) -> bool:
        return super().ready() and bool(self._score) and self._sensor_count > 0

    def preempted(self) -> bool:
        return self._preempt

    def goto(  # type: ignore[override]
        self,
        x: float,
        y: float,
        timeout: float = 180.0,
        guard=None,
    ) -> NavResult:
        result = super().goto(x, y, timeout, guard)
        return NavResult(
            str(result.get('status', 'failed')),
            str(result.get('reason', '')),
            float(result.get('distance_to_goal', 0.0)),
            int(result.get('replans', 0)),
        )

    def read_sensor(self, count: int = 5) -> Reading:
        self.stop()
        values: list[float] = []
        last = self._sensor_count
        deadline = time.monotonic() + 10.0
        while len(values) < count and time.monotonic() < deadline:
            rclpy.spin_once(self, timeout_sec=0.1)
            if self._sensor_count != last:
                last = self._sensor_count
                values.append(self._sensor)
        if not values:
            return Reading(self._sensor, 0.0)
        mid = median(values)
        return Reading(mid, 1.4826 * median(abs(v - mid) for v in values))

    def _call(self, client) -> tuple[bool, str]:
        if not client.wait_for_service(timeout_sec=5.0):
            return False, 'service unavailable'
        future = client.call_async(Trigger.Request())
        rclpy.spin_until_future_complete(self, future, timeout_sec=5.0)
        result = future.result()
        if result is None:
            return False, 'service call timed out'
        return bool(result.success), str(result.message)

    def collect(self) -> tuple[bool, str]:
        ok, message = self._call(self._collect)
        if ok:  # /did/score lags a tick behind the service reply
            self._score['collected'] = self.collected() + 1
        return ok, message

    def finish(self) -> tuple[bool, str]:
        return self._call(self._finish)

    def noise_level(self) -> float:
        return self.adaptation.noise_level()

    def anomaly(self) -> dict[str, bool]:
        return self.monitor.anomaly(self.now())

    def battery(self) -> float:
        return self._battery

    def samples_total(self) -> int:
        return int(self._score.get('samples_total', 0))

    def collected(self) -> int:
        return int(self._score.get('collected', 0))

    # --- publishing -------------------------------------------------------------------------

    def journal(self, kind: str, title: str, text: str = '', status: str = 'open') -> None:
        """Write an entry to the experiment journal shown on the dashboard."""
        entry = {'t': round(self.now(), 1), 'kind': kind, 'title': title,
                 'text': text, 'status': status}
        self._journal_pub.publish(String(data=json.dumps(entry, ensure_ascii=False)))

    def _publish_costmap(self) -> None:
        signature = (self.costmap.version, self.adaptation.learner.observations)
        if signature == self._published_cost_signature:
            return
        self._published_cost_signature = signature
        payload = costmap_layers(self.costmap)
        payload['knowledge_version'] = signature[1]
        self._costmap_pub.publish(String(data=json.dumps(payload)))

    def _publish_status(self, status: dict[str, Any]) -> None:
        self._current = status
        self._status_pub.publish(String(data=json.dumps(status)))

    def _publish_state(self) -> None:
        pose = self.pose()
        if pose is None:
            return
        t = self.now()
        try:
            return_cost = self.skills.return_cost_estimate()
        except Exception:  # noqa: BLE001 - the state topic must never crash the node
            return_cost = float('nan')
        state = {
            't': round(t, 2),
            'pose': {'x': round(pose.x, 3), 'y': round(pose.y, 3), 'yaw': round(pose.yaw, 3)},
            'battery': round(self._battery, 3),
            'collected': self.collected(),
            'samples_total': self.samples_total(),
            'score': self._score.get('score'),
            'sensor': {
                'value': round(self._sensor, 3),
                'noise_estimate': round(self.monitor.noise_estimate, 4),
            },
            'current': {
                'plan_id': self._current.get('plan_id', ''),
                'index': self._current.get('index', 0),
                'type': self._current.get('type', ''),
                'state': self._current.get('state', 'idle'),
            },
            'recent_events': self.monitor.recent_events(),
            'return_cost_estimate': None if return_cost != return_cost else round(return_cost, 2),
            'anomaly': self.monitor.anomaly(t),
            'cost_map_updates': self.cost_log,
            'navigation': {
                'status': self.core.status,
                'replans': self.core.replans,
                'waypoints': [[round(x, 2), round(y, 2)] for x, y in self.core.waypoints[:40]]
                if self.core.status == 'running' else [],
            },
        }
        self._state_pub.publish(String(data=json.dumps(state)))
        self._publish_costmap()

    # --- main loops ----------------------------------------------------------------------------

    def serve_plans(self) -> None:
        """Wait for plans and commands and execute them one after another."""
        self.get_logger().info('waiting for plans on /agent/plan and commands on /agent/command')
        while rclpy.ok():
            rclpy.spin_once(self, timeout_sec=0.1)
            if self._scenario_reset is not None:
                self._reset_for_scenario(self._scenario_reset)
            elif self._command == 'auto':
                self._command, self._preempt = None, False
                self._run_auto_with_status()
            elif self._pending is not None:
                plan, self._pending, self._preempt = self._pending, None, False
                if not self.ready():
                    self.wait_for_sensors()
                self.plan_executor.run(plan)

    def _run_auto_with_status(self) -> None:
        status = {'plan_id': 'auto', 'index': 0, 'type': 'auto',
                  'subgoal': 'автономный режим', 'state': 'running', 'reason': '', 'data': {}}
        self._publish_status(status)
        self.journal('decision', 'Запуск автономного режима',
                     'Обхожу арену, по сигналу датчика ищу образцы, слежу за запасом батареи.')
        summary = self.run_autonomous()
        state = 'preempted' if self._preempt else (
            'done' if summary['returned_to_base'] else 'failed')
        self._publish_status({**status, 'state': state,
                              'reason': summary.get('reason', ''), 'data': summary})
        self.journal('result', 'Эпизод завершён',
                     f"Собрано {summary['collected']} из {summary['samples_total']}, "
                     f"батарея {summary['battery']}, "
                     f"{'на базе' if summary['returned_to_base'] else 'не вернулся на базу'}",
                     'confirmed' if summary['returned_to_base'] else 'rejected')

    def run_autonomous(self) -> dict[str, Any]:
        """Run the fallback planner without an LLM and return its summary."""
        self.wait_for_sensors()

        def log(message: str) -> None:
            self.get_logger().info(message)
            self.journal('agent', message)

        return AutonomousAgent(self, self.skills, log=log).run()


def main(args=None) -> None:
    """``ros2 run did_agent agent``: execute plans sent to /agent/plan."""
    rclpy.init(args=args)
    node = AgentNode('agent')
    try:
        node.serve_plans()
    except KeyboardInterrupt:
        pass
    finally:
        node.stop()
        node.destroy_node()
        rclpy.shutdown()


def main_auto(args=None) -> None:
    """``ros2 run did_agent auto``: run the autonomous agent once and exit."""
    rclpy.init(args=args)
    node = AgentNode('auto_agent')
    code = 1
    try:
        summary = node.run_autonomous()
        print(json.dumps(summary, sort_keys=True))
        code = 0 if summary['returned_to_base'] else 1
    except KeyboardInterrupt:
        pass
    finally:
        node.stop()
        node.destroy_node()
        rclpy.shutdown()
    raise SystemExit(code)


def main_send_plan(args=None) -> None:
    """``ros2 run did_agent send_plan '<json>' [--wait]``: publish a plan once."""
    parser = argparse.ArgumentParser(description='Publish a plan to /agent/plan.')
    parser.add_argument('plan', help='plan JSON text')
    parser.add_argument('--wait', action='store_true', help='wait until it finishes')
    parser.add_argument('--timeout', type=float, default=600.0)
    options = parser.parse_args(remove_ros_args(args if args is not None else sys.argv)[1:])

    rclpy.init(args=args)
    node = rclpy.create_node('send_plan')
    publisher = node.create_publisher(String, '/agent/plan', 10)
    states: list[dict[str, Any]] = []
    node.create_subscription(
        String, '/agent/status', lambda m: states.append(json.loads(m.data)), 50
    )
    code = 0
    try:
        parsed = parse_plan(options.plan)
    except PlanError as error:
        print(f'invalid plan: {error}', file=sys.stderr)
        raise SystemExit(2)
    deadline = time.monotonic() + 15.0
    while publisher.get_subscription_count() == 0 and time.monotonic() < deadline:
        rclpy.spin_once(node, timeout_sec=0.1)
    publisher.publish(String(data=options.plan))
    if options.wait:
        last = len(parsed.subgoals) - 1
        deadline = time.monotonic() + options.timeout
        final = None
        while rclpy.ok() and time.monotonic() < deadline and final is None:
            rclpy.spin_once(node, timeout_sec=0.2)
            for status in states:
                if status.get('plan_id') != parsed.plan_id:
                    continue
                if status['state'] in ('failed', 'preempted') or (
                    status['state'] == 'done' and status['index'] == last
                ):
                    final = status
        print(json.dumps(final or {'state': 'timeout'}, sort_keys=True))
        code = 0 if final and final['state'] == 'done' else 1
    node.destroy_node()
    rclpy.shutdown()
    raise SystemExit(code)


def main_command(args=None) -> None:
    """``ros2 run did_agent command auto|stop [--wait]``: send a command to the agent."""
    parser = argparse.ArgumentParser(description='Send a command to /agent/command.')
    parser.add_argument('cmd', choices=['auto', 'stop'])
    parser.add_argument('--wait', action='store_true', help='wait for the autonomous run to end')
    parser.add_argument('--timeout', type=float, default=900.0)
    options = parser.parse_args(remove_ros_args(args if args is not None else sys.argv)[1:])

    rclpy.init(args=args)
    node = rclpy.create_node('send_command')
    publisher = node.create_publisher(String, '/agent/command', 10)
    seen: list[dict[str, Any]] = []
    node.create_subscription(
        String, '/agent/status', lambda m: seen.append(json.loads(m.data)), 50
    )
    deadline = time.monotonic() + 15.0
    while publisher.get_subscription_count() == 0 and time.monotonic() < deadline:
        rclpy.spin_once(node, timeout_sec=0.1)
    publisher.publish(String(data=json.dumps({'cmd': options.cmd})))
    code = 0
    if options.wait and options.cmd == 'auto':
        final = None
        deadline = time.monotonic() + options.timeout
        while rclpy.ok() and time.monotonic() < deadline and final is None:
            rclpy.spin_once(node, timeout_sec=0.3)
            for status in seen:
                if status.get('plan_id') == 'auto' and status['state'] in (
                    'done', 'failed', 'preempted'
                ):
                    final = status
        result = (final or {}).get('data') or {'state': 'timeout'}
        print(json.dumps(result, sort_keys=True))
        code = 0 if final and final['state'] == 'done' else 1
    node.destroy_node()
    rclpy.shutdown()
    raise SystemExit(code)
