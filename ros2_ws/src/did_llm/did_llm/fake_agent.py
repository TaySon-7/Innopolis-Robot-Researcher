#!/usr/bin/env python3
"""A stand-in for the agent, for exercising the planner without Gazebo.

The planner's contract is two messages in (``/agent/state`` once a second and
``/agent/status``) and two out (``/agent/plan``, and ``/agent/command`` only as
a fallback). This node plays the agent's side of that contract, so the planner
can be run end to end without the simulator.

WHAT THIS IS NOT
----------------
It is not the agent, and not a measurement. The planner does not depend on the
robot, only on the messages, so a stand-in exercises the planner honestly:
same formats, same cadence, same rejection paths. What it cannot tell anyone is
whether the plans are good for the robot, because it does not drive one.

Geometry and events here are set by scenario, not by physics. Every field it
publishes has the shape and units of the real one, and nothing more.
"""

from __future__ import annotations

import json
from typing import Any

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile
from std_msgs.msg import String

STATE_TOPIC = '/agent/state'
STATUS_TOPIC = '/agent/status'
PLAN_TOPIC = '/agent/plan'
COMMAND_TOPIC = '/agent/command'

BASE_X, BASE_Y = -2.0, -0.5

#: Hidden samples, as a scenario YAML would place them.
SAMPLES = [
    {'id': 's1', 'x': -1.5, 'y': -0.5},
    {'id': 's2', 'x': 0.5, 'y': 1.5},
    {'id': 's3', 'x': 1.8, 'y': -1.8},
]

SOIL_ZONES = [
    {'id': 'z1', 'circle': {'x': 0.0, 'y': -0.5, 'r': 0.4}, 'cost': 2.5},
]

HAZARD_ZONES = [
    {'id': 'h1', 'circle': {'x': 1.0, 'y': 1.0, 'r': 0.3}},
]

BASE_COST_PER_M = 1.0
COLLECT_RADIUS = 0.30
SENSOR_RANGE = 1.5
SENSOR_NOISE = 0.01


def _inside(x: float, y: float, shape: dict[str, Any]) -> bool:
    cx, cy, r = shape['x'], shape['y'], shape['r']
    return (x - cx) ** 2 + (y - cy) ** 2 <= r * r


def drain_at(x: float, y: float) -> float:
    """Battery per metre at a point."""
    for zone in SOIL_ZONES:
        if _inside(x, y, zone['circle']):
            return BASE_COST_PER_M * zone['cost']
    return BASE_COST_PER_M


class FakeAgent(Node):
    """Publishes ``/agent/state`` and executes what the planner sends."""

    def __init__(self) -> None:
        super().__init__('fake_agent')

        qos = QoSProfile(depth=10)
        self.state_pub = self.create_publisher(String, STATE_TOPIC, qos)
        self.status_pub = self.create_publisher(String, STATUS_TOPIC, qos)
        self.score_pub = self.create_publisher(String, '/did/score', qos)
        self.create_subscription(String, PLAN_TOPIC, self._on_plan, qos)
        self.create_subscription(String, COMMAND_TOPIC, self._on_command, qos)

        self.battery = 60.0
        self.pose = {'x': BASE_X, 'y': BASE_Y, 'yaw': 0.0}
        self.collected = 0
        self.score = 0
        self.collected_ids: set[str] = set()
        self.distance = 0.0
        self.ticks = 0
        self.cost_log: list[dict[str, Any]] = []
        self.anomaly = {'battery_deviation': False,
                        'penalties_burst': False,
                        'sensor_noise_up': False}
        self.navigation = {'status': 'idle', 'replans': 0}

        self.executing = False
        self.current_index = 0
        self.current = {'plan_id': '', 'index': 0, 'type': '',
                        'state': 'idle'}
        self.command: str | None = None
        self.finished = False

        self.create_timer(1.0, self._publish_state)

    # ------------------------------------------------------------------ input
    def _on_command(self, message: String) -> None:
        try:
            command = json.loads(message.data).get('cmd')
        except (TypeError, ValueError, AttributeError):
            command = None
        self.get_logger().info(f'команда: {command!r}')
        if command == 'auto':
            self.command = 'auto'
            self.get_logger().info('принят автономный режим')
        elif command == 'stop':
            self.command = None
            self.finished = True

    def _on_plan(self, message: String) -> None:
        """Validate like the executor does, then report what it decided.

        The point of the stand-in is the failure paths, so it rejects exactly
        what ``did_agent.plan.parse_plan`` rejects and says so in the same
        words.
        """
        plan = self._parse(message.data)
        if isinstance(plan, str):
            self._status(state='failed',
                         reason=f'invalid plan: {plan}', plan_id='')
            self.get_logger().warning(f'план отклонён: {plan}')
            return

        self.get_logger().info(
            f'план {plan["plan_id"]}: '
            f'{[item["type"] for item in plan["subgoals"]]}'
        )
        self.current_plan = plan
        self.current_plan['index'] = 0
        self.command = None
        # The stand-in does not move; it walks the plan so the planner sees the
        # same status sequence the real executor produces.
        self._advance_plan()

    def _parse(self, text: str) -> Any:
        fields = {'goto': ('x', 'y'),
                  'search_around': ('x', 'y', 'radius'),
                  'collect': (), 'return_to_base': ()}
        try:
            data = json.loads(text)
        except (TypeError, ValueError) as error:
            return f'plan is not valid JSON: {error}'
        if not isinstance(data, dict) or not isinstance(data.get('subgoals'), list):
            return 'plan has no "subgoals" list'
        for index, item in enumerate(data['subgoals']):
            if not isinstance(item, dict):
                return f'subgoal {index}: must be an object'
            kind = item.get('type')
            if kind not in fields:
                return (f'subgoal {index}: unknown type {kind!r} '
                        f"(known: {', '.join(fields)})")
            for key in fields[kind]:
                value = item.get(key)
                if isinstance(value, bool) or not isinstance(value, (int, float)):
                    return (f'subgoal {index}: field {key!r} must be a number, '
                            f'got {value!r}')
            if kind == 'search_around' and not 0.1 <= item['radius'] <= 3.0:
                return f'subgoal {index}: radius must be between 0.1 and 3.0 m'
        return data

    def _advance_plan(self) -> None:
        """Finish the current plan: one subgoal per tick."""
        subgoals = self.current_plan['subgoals']
        index = self.current_plan['index']
        if index >= len(subgoals):
            self._status(state='done', reason='',
                         plan_id=self.current_plan['plan_id'], type='')
            return

        subgoal = subgoals[index]
        kind = subgoal['type']
        self.current_index = index + 1
        self.current = {'plan_id': self.current_plan['plan_id'],
                        'index': index + 1, 'type': kind, 'state': 'running'}
        self._status(state='running', reason='', type=kind,
                     plan_id=self.current_plan['plan_id'],
                     subgoal=f'{kind}')

        if kind == 'goto':
            self.pose = {'x': subgoal['x'], 'y': subgoal['y'], 'yaw': 0.0}
        elif kind == 'search_around':
            self.pose = {'x': subgoal['x'], 'y': subgoal['y'], 'yaw': 0.0}
        elif kind == 'collect':
            self._try_collect()
        elif kind == 'return_to_base':
            self.pose = {'x': BASE_X, 'y': BASE_Y, 'yaw': 0.0}
            if self.collected >= len(SAMPLES):
                self.finished = True
                self.score += 5

        # The pose moved, so pay for the distance at the new ground's price.
        self.battery = max(0.0, self.battery - self._last_cost())

        if self.current_plan['index'] + 1 >= len(subgoals):
            self._status(state='done', reason='', type=kind,
                         plan_id=self.current_plan['plan_id'])
        else:
            self.current_plan['index'] += 1

    def _try_collect(self) -> None:
        for sample in SAMPLES:
            if sample['id'] in self.collected_ids:
                continue
            if ((self.pose['x'] - sample['x']) ** 2
                    + (self.pose['y'] - sample['y']) ** 2) ** 0.5 < COLLECT_RADIUS:
                self.collected_ids.add(sample['id'])
                self.collected += 1
                self.score += 10
                return
        self._status(state='failed', reason='no sample within radius',
                     type='collect', plan_id=self.current_plan['plan_id'])

    def _last_cost(self) -> float:
        return 0.5 * drain_at(self.pose['x'], self.pose['y'])

    def _status(self, *, state: str, reason: str, plan_id: str = '',
                 type: str = '', subgoal: str = '') -> None:
        message = String(data=json.dumps({
            'plan_id': plan_id,
            'index': self.current_index,
            'type': type,
            'subgoal': subgoal or type,
            'state': state,
            'reason': reason,
            'data': {},
        }, ensure_ascii=False))
        self.status_pub.publish(message)

    # ----------------------------------------------------------------- output
    def _publish_state(self) -> None:
        signal, noise = self._read_sensor()
        # Exactly the shape did_agent.agent_node publishes: /agent/state is the
        # state object itself, and its "score" field is a number. The episode
        # flag lives in the judge's /did/score, not here.
        state = {
            't': 0.0,
            'scenario': 'easy',
            'pose': {key: round(value, 3)
                     for key, value in self.pose.items()},
            'battery': round(self.battery, 3),
            'collected': self.collected,
            'samples_total': len(SAMPLES),
            'score': self.score,
            'sensor': {'value': round(signal, 3),
                       'noise_estimate': round(noise, 4)},
            'current': dict(self.current),
            'recent_events': [],
            'return_cost_estimate': round(self._return_cost(), 2),
            'anomaly': dict(self.anomaly),
            'cost_map_updates': list(self.cost_log[-10:]),
            'navigation': dict(self.navigation),
        }
        self.state_pub.publish(String(data=json.dumps(state)))

        # The judge's score topic, which is where "finished" is published.
        # Like the real judge it never carries uncollected sample positions
        # or a world pose: those are hidden truth (see did_judge.score_payload).
        self.score_pub.publish(String(data=json.dumps({
            'scenario': 'easy',
            'battery': round(self.battery, 3),
            'collected': self.collected,
            'samples_total': len(SAMPLES),
            'distance_travelled': self.distance,
            'finished': self.finished,
        }, sort_keys=True)))

        if self.current.get('state') == 'running':
            self._advance_plan()

    def _return_cost(self) -> float:
        distance = ((self.pose['x'] - BASE_X) ** 2
                    + (self.pose['y'] - BASE_Y) ** 2) ** 0.5
        return distance * BASE_COST_PER_M

    def _read_sensor(self) -> tuple[float, float]:
        best = 0.0
        for sample in SAMPLES:
            if sample['id'] in self.collected_ids:
                continue
            distance = ((self.pose['x'] - sample['x']) ** 2
                        + (self.pose['y'] - sample['y']) ** 2) ** 0.5
            best = max(best, max(0.0, 1.0 - distance / SENSOR_RANGE))
        # The judge's sensor carries noise; so does this one, fixed, so the
        # planner sees the same kind of jitter rather than a clean signal.
        jitter = ((self.ticks % 7) - 3) * SENSOR_NOISE / 3
        self.ticks += 1
        return max(0.0, min(1.0, best + jitter)), SENSOR_NOISE


def main(args=None) -> None:
    rclpy.init(args=args)
    node = FakeAgent()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
