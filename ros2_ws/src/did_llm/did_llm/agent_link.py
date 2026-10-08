"""ROS wiring for the planner: what it reads and what it publishes.

Reads only what the executor offers on ``/agent/*``. The judge is not
subscribed to at all: the agent has already folded battery, sensor and events
into ``/agent/state``, and reading the judge directly would tie the planner to
this particular judge instead of to the contract.
"""

from __future__ import annotations

import json
from typing import Any, Callable

import rclpy
from rclpy.node import Node
from std_msgs.msg import String

PLAN_TOPIC = '/agent/plan'
STATE_TOPIC = '/agent/state'
STATUS_TOPIC = '/agent/status'
COMMAND_TOPIC = '/agent/command'
JOURNAL_TOPIC = '/agent/journal'

#: The judge publishes the episode flag here and nowhere else: inside
#: /agent/state the "score" field is a plain number. Reading this one public
#: judge topic is the whole of the planner's contact with the judge.
SCORE_TOPIC = '/did/score'


class AgentLink:
    """A thin transport over the executor's interface.

    Not a Node of its own: it registers its publishers and subscriptions on
    the node that owns it. Two nodes sharing a name in one process split the
    rosout logger between them and make the logs misleading.
    """

    def __init__(self, node: Node) -> None:
        self.node = node
        #: Optional: the owner node's logger, for transport-level notes.
        self.log = node.get_logger()

        self.plan_pub = node.create_publisher(String, PLAN_TOPIC, 10)
        self.command_pub = node.create_publisher(String, COMMAND_TOPIC, 10)
        self.journal_pub = node.create_publisher(String, JOURNAL_TOPIC, 10)

        self.state: dict[str, Any] | None = None
        self.state_at: float | None = None
        self.status: dict[str, Any] | None = None
        self.seen_plans: set[str] = set()
        #: From the judge: whether the episode is over.
        self.episode_finished = False

        node.create_subscription(String, STATE_TOPIC, self._on_state, 10)
        node.create_subscription(String, STATUS_TOPIC, self._on_status, 10)
        node.create_subscription(String, SCORE_TOPIC, self._on_score, 10)

    # ------------------------------------------------------------------ input
    def _on_state(self, message: String) -> None:
        try:
            payload = json.loads(message.data)
        except (TypeError, ValueError):
            return
        if not isinstance(payload, dict):
            return
        self.state = payload
        self.state_at = self.now()

    def _on_status(self, message: String) -> None:
        try:
            payload = json.loads(message.data)
        except (TypeError, ValueError):
            return
        if not isinstance(payload, dict):
            return
        self.status = payload

    def _on_score(self, message: String) -> None:
        """The only thing taken from the judge: whether the episode is over."""
        try:
            payload = json.loads(message.data)
        except (TypeError, ValueError):
            return
        if isinstance(payload, dict):
            self.episode_finished = bool(payload.get('finished', False))

    def now(self) -> float:
        """Clock reading shared by the node, so timings stay comparable."""
        return float(self.node.get_clock().now().nanoseconds) / 1e9

    # ----------------------------------------------------------------- output
    def state_age(self) -> float | None:
        if self.state_at is None:
            return None
        return self.now() - self.state_at

    def publish_plan(self, payload: dict[str, Any]) -> None:
        self.plan_pub.publish(
            String(data=json.dumps(payload, ensure_ascii=False)))

    def publish_command(self, command: str) -> None:
        self.command_pub.publish(
            String(data=json.dumps({'cmd': command})))

    def journal(self, kind: str, title: str, text: str = '',
                status: str = 'open', **extra: Any) -> None:
        """One entry of the experiment journal, as the dashboard reads it.

        ``kind`` is one of hypothesis, result, decision, agent; ``status`` one
        of open, confirmed, rejected. The planner uses them as documented
        rather than inventing a schema, so entries show up in the dashboard's
        feed unchanged.
        """
        record = {
            't': round(self.now(), 2),
            'kind': kind,
            'status': status,
            'title': title,
            'text': text,
            **extra,
        }
        self.journal_pub.publish(
            String(data=json.dumps(record, ensure_ascii=False)))

    # ------------------------------------------------------------------ state
    def finished(self) -> bool:
        """True once the judge says the episode is over.

        ``/agent/state`` carries a numeric score, so the flag cannot be read
        from there; it comes from the judge.
        """
        return self.episode_finished

    def battery(self) -> float:
        return float((self.state or {}).get('battery', 0.0))

    def return_cost(self) -> float | None:
        value = (self.state or {}).get('return_cost_estimate')
        return float(value) if isinstance(value, (int, float)) else None

    def remaining_samples(self) -> int:
        state = self.state or {}
        return max(0, int(state.get('samples_total', 0))
                   - int(state.get('collected', 0)))

    def last_status_for(self, plan_id: str) -> dict[str, Any] | None:
        """The outcome of a plan we published, once it is known."""
        status = self.status
        if not status or status.get('plan_id') != plan_id:
            return None
        if status.get('state') not in ('done', 'failed', 'preempted'):
            return None
        return status