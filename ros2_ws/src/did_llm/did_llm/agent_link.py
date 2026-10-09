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

#: The hypothesis book as a table, published alongside the prose journal.
HYPOTHESIS_TOPIC = '/agent/hypotheses'

#: The agent's own cost map. It is the only place the expensive ground the
#: learner has measured actually appears: ``/agent/state`` carries an empty
#: ``cost_map_updates`` until an analyst pushes something, so a planner that
#: reads only state has no idea where the floor is dear.
COSTMAP_TOPIC = '/agent/costmap'

#: Grid the cost map is expressed in, from /api/geometry.
GRID_ORIGIN = (-10.0, -10.0)
GRID_STEP = 0.05

#: Floor cost above which the agent itself treats ground as expensive.
EXPENSIVE_COST = 1.4

#: The judge publishes the episode flag here and nowhere else: inside
#: /agent/state the "score" field is a plain number. Reading this one public
#: judge topic is the whole of the planner's contact with the judge.
SCORE_TOPIC = '/did/score'
EVENTS_TOPIC = '/did/events'


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
        self.hypothesis_pub = node.create_publisher(String, HYPOTHESIS_TOPIC, 10)

        self.state: dict[str, Any] | None = None
        self.state_at: float | None = None
        self.status: dict[str, Any] | None = None
        self.seen_plans: set[str] = set()
        #: The operator's last command, and whether the planner has seen it
        #: yet. ``stop`` has to be obeyed, and nothing else in the status
        #: stream says the operator asked for it.
        self.command: str | None = None
        self.command_pending = False
        #: Subgoal reports already logged, so the log shows progress rather
        #: than the same line repeated at the topic's publish rate.
        self._logged_status: set[tuple[str, Any, str]] = set()
        #: From the judge: whether the episode is over.
        self.episode_finished = False
        #: Monotonic local counter incremented whenever /agent/state announces
        #: a new episode. The explicit id also catches a restart of the same
        #: scenario while the global Gazebo clock keeps advancing.
        self.episode_generation = 0
        self._agent_episode_id: int | None = None
        #: Expensive ground from the agent's cost map: (x, y, reach, cost).
        self.expensive: list[tuple[float, float, float, float]] = []
        #: Penalty events waiting to be acted on.
        self.pending_events: list[dict[str, Any]] = []
        #: Latest collection count seen on the event stream, which arrives
        #: before the next /agent/state does.
        self.collected_hint: int | None = None

        node.create_subscription(String, STATE_TOPIC, self._on_state, 10)
        node.create_subscription(String, STATUS_TOPIC, self._on_status, 10)
        node.create_subscription(String, SCORE_TOPIC, self._on_score, 10)
        node.create_subscription(String, EVENTS_TOPIC, self._on_event, 50)
        node.create_subscription(String, COSTMAP_TOPIC, self._on_costmap, 10)
        node.create_subscription(String, COMMAND_TOPIC, self._on_command, 10)

    # ------------------------------------------------------------------ input
    def _on_command(self, message: String) -> None:
        """An operator command from the dashboard's Стоп / Автономно buttons.

        Watched because it cannot be inferred from ``/agent/status``: a stop
        cancels the goal without starting a plan, so the status still carries
        this planner's own plan id and there is nothing there to notice.
        """
        try:
            payload = json.loads(message.data)
        except (TypeError, ValueError):
            return
        if not isinstance(payload, dict):
            return
        command = payload.get('cmd')
        if isinstance(command, str) and command:
            self.command = command
            self.command_pending = True

    def _on_state(self, message: String) -> None:
        try:
            payload = json.loads(message.data)
        except (TypeError, ValueError):
            return
        if not isinstance(payload, dict):
            return

        raw_episode_id = payload.get('episode_id')
        episode_id = (
            raw_episode_id
            if isinstance(raw_episode_id, int) and not isinstance(raw_episode_id, bool)
            else None
        )
        if episode_id is not None:
            if (
                self._agent_episode_id is not None
                and episode_id != self._agent_episode_id
            ):
                self.episode_generation += 1
                # Discard only previous-run observations. Keep the state below:
                # it is already the first clean snapshot of the new episode.
                self.status = None
                self.seen_plans.clear()
                self._logged_status.clear()
                self.pending_events.clear()
                self.collected_hint = None
                self.expensive.clear()
                self.episode_finished = False
            self._agent_episode_id = episode_id
        self.state = payload
        self.state_at = self.now()

    def _on_status(self, message: String) -> None:
        """Every subgoal report arrives here: the agent's side of the wire.

        Logged, because "what did the robot actually do with the plan" was
        invisible: the planner only saw the final outcome, so a plan that
        stalled on one subgoal looked the same as one that sailed through all
        of them. Only a change of (plan, subgoal, state) is logged, so the
        running/running/running updates do not repeat.
        """
        try:
            payload = json.loads(message.data)
        except (TypeError, ValueError):
            return
        if not isinstance(payload, dict):
            return
        self.status = payload

        marker = (str(payload.get('plan_id') or ''),
                  payload.get('index'),
                  str(payload.get('state') or ''))
        if marker in self._logged_status:
            return
        # The set only grows with distinct (plan, subgoal, state) triples, so
        # it is trimmed rather than reset: a reset would re-log whatever the
        # agent is still repeating.
        if len(self._logged_status) > 512:
            self._logged_status.clear()
        self._logged_status.add(marker)

        reason = str(payload.get('reason') or '')
        detail = payload.get('data')
        extra = f'   {detail}' if isinstance(detail, dict) and detail else ''
        self.log.info(
            f'АГЕНТ → БУРГЕР   {marker[0]}  подцель {marker[1]}: '
            f'{payload.get("type", "?")}({payload.get("subgoal", "?")})'
            f'  →  {marker[2]}'
            + (f'  причина: {reason}' if reason else '')
            + extra)

    def _on_event(self, message: String) -> None:
        """Penalty events, drained on the next tick rather than acted on here.

        The callback must stay cheap and must not publish: a stop and a fresh
        model call belong to the planning loop, which is the only place that
        knows what is already running.
        """
        try:
            payload = json.loads(message.data)
        except (TypeError, ValueError):
            return
        if not isinstance(payload, dict):
            return
        name = payload.get('event')
        if isinstance(name, str) and name:
            self.pending_events.append(payload)
            if name == 'sample_collected':
                collected = payload.get('collected')
                if isinstance(collected, (int, float)):
                    self.collected_hint = int(collected)

    def take_event(self) -> dict[str, Any] | None:
        """The oldest unhandled event, if any."""
        return self.pending_events.pop(0) if self.pending_events else None

    def note_collected(self, value: int) -> None:
        """A collection reported by an event, ahead of the next state."""
        self.collected_hint = value

    def _on_score(self, message: String) -> None:
        """Track the judge's public episode completion flag."""
        try:
            payload = json.loads(message.data)
        except (TypeError, ValueError):
            return
        if not isinstance(payload, dict):
            return

        self.episode_finished = bool(payload.get('finished', False))

    def _on_costmap(self, message: String) -> None:
        """Read the expensive ground the agent has measured.

        The agent encodes it as ``[row, col_from, col_to, cost]`` runs on a
        0.05 m grid with row 0 at the bottom. Each run becomes a world-space
        point with a reach, which is the granularity a plan needs: a plan names
        coordinates, so it needs coordinates back.

        Without this the planner is blind. ``/agent/state`` carries an empty
        ``cost_map_updates`` until an analyst pushes something, so a planner
        that reads only state cannot route around expensive ground and will
        happily drive the robot across it.
        """
        try:
            payload = json.loads(message.data)
        except (TypeError, ValueError):
            return
        runs = payload.get('terrain') if isinstance(payload, dict) else None
        if not isinstance(runs, list):
            return

        regions: list[tuple[float, float, float, float]] = []
        for run in runs:
            try:
                row, col0, col1, cost = (float(value) for value in run[:4])
            except (TypeError, ValueError):
                continue
            if cost < EXPENSIVE_COST:
                continue
            x = GRID_ORIGIN[0] + (col0 + col1) / 2 * GRID_STEP
            y = GRID_ORIGIN[1] + row * GRID_STEP
            # Half the run's width, so a long strip is not reported as a point
            # the plan can simply walk around.
            reach = max(GRID_STEP, (col1 - col0 + 1) * GRID_STEP / 2)
            regions.append((x, y, reach, cost))
        self.expensive = regions

    def expensive_ground(self) -> list[dict[str, float]]:
        """Expensive patches, for the prompt."""
        return [{'x': round(x, 2), 'y': round(y, 2),
                 'reach': round(reach, 2), 'cost': round(cost, 2)}
                for x, y, reach, cost in self.expensive]

    def cost_at(self, x: float, y: float) -> float:
        """Measured floor cost at a point; 1.0 where nothing is known."""
        worst = 1.0
        for px, py, reach, cost in self.expensive:
            if (x - px) ** 2 + (y - py) ** 2 <= reach ** 2 and cost > worst:
                worst = cost
        return worst

    def now(self) -> float:
        """Clock reading shared by the node, so timings stay comparable."""
        return float(self.node.get_clock().now().nanoseconds) / 1e9

    # ----------------------------------------------------------------- output
    def state_age(self) -> float | None:
        if self.state_at is None:
            return None
        return self.now() - self.state_at

    def publish_plan(self, payload: dict[str, Any]) -> None:
        plan_id = str(payload.get('plan_id') or '')
        if plan_id:
            self.seen_plans.add(plan_id)
        self.plan_pub.publish(
            String(data=json.dumps(payload, ensure_ascii=False)))

    def publish_command(self, command: str) -> None:
        self.command_pub.publish(
            String(data=json.dumps({'cmd': command})))

    def publish_hypotheses(self, hypotheses: list[dict[str, Any]]) -> None:
        """Publish the hypothesis book, for the dashboard's own panel.

        Separate from the journal because the journal is prose and this is the
        table: every claim with its rule, its status and the evidence behind it.
        A dashboard that has to reassemble the table out of sentences cannot
        show a count, and a count is the part a viewer checks first.
        """
        self.hypothesis_pub.publish(
            String(data=json.dumps({'hypotheses': hypotheses},
                                   ensure_ascii=False)))

    def journal(self, kind: str, title: str, text: str = '',
                status: str = 'open', **extra: Any) -> None:
        """One entry of the experiment journal, as the dashboard reads it.

        ``kind`` identifies the source (for example ``llm`` or ``robot``), and
        ``status`` is one of open, confirmed, rejected. Entries pass through
        unchanged so the dashboard can label and colour each source.
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

    def pose(self) -> tuple[float, float] | None:
        pose = (self.state or {}).get('pose')
        if isinstance(pose, dict) and isinstance(pose.get('x'), (int, float)) \
                and isinstance(pose.get('y'), (int, float)):
            return float(pose['x']), float(pose['y'])
        return None

    def signal(self) -> float | None:
        """Sensor reading: 0..1, how close the nearest sample is.

        The judge publishes no direction, only magnitude. That is why a plan
        must search where it stands when this is high rather than set off for
        a guessed point.
        """
        sensor = (self.state or {}).get('sensor')
        if isinstance(sensor, dict) and isinstance(sensor.get('value'),
                                                   (int, float)):
            return float(sensor['value'])
        return None

    def noise(self) -> float | None:
        sensor = (self.state or {}).get('sensor')
        if isinstance(sensor, dict) and isinstance(
                sensor.get('noise_estimate'), (int, float)):
            return float(sensor['noise_estimate'])
        return None

    def return_cost(self) -> float | None:
        value = (self.state or {}).get('return_cost_estimate')
        return float(value) if isinstance(value, (int, float)) else None

    def in_autonomous_mode(self) -> bool:
        """Whether the agent is driving itself, by operator or by fallback.

        The agent reports ``plan_id: "auto"`` for its own behaviour. While
        that holds, any plan we publish would preempt it — and during a demo
        the operator may press the Autonomous button at any moment, so the
        planner has to notice rather than keep fighting for control.
        """
        return str((self.state or {}).get('current', {}).get('plan_id', '')) == 'auto'

    def remaining_samples(self) -> int:
        state = self.state or {}
        collected = state.get('collected', 0)
        # The event stream reports a collection the instant it happens, the
        # state snapshot only once a second. Taking the larger count keeps a
        # plan from being written for a sample that is already in the bag.
        if self.collected_hint is not None:
            collected = max(int(collected), self.collected_hint)
        return max(0, int(state.get('samples_total', 0)) - int(collected))

    def last_status_for(self, plan_id: str) -> dict[str, Any] | None:
        """The outcome of a plan we published, once it is known."""
        status = self.status
        if not status or status.get('plan_id') != plan_id:
            return None
        if status.get('state') not in ('done', 'failed', 'preempted'):
            return None
        return status
