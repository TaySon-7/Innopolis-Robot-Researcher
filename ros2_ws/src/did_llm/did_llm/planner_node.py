"""The planning loop: state in, plan out.

The order of the checks matters more than the model call. Before asking
anything, the loop settles what can be decided without a model: the episode is
over, the budget forbids going out, or the API is down. Each of those has a
deterministic answer, and spending a model call to rediscover it would make
the planner both slower and less reliable.
"""

from __future__ import annotations

import json
from math import atan2, cos, hypot, radians, sin
import threading
from typing import Any

from did_llm.agent_plan import (
    ARENA,
    COLLISION_RADIUS_M,
    GOTO_NOOP_M,
    PILLAR_KEEPOUT,
    SEARCH_WORTH_IT,
    SIGNAL_CLOSE,
    SIGNAL_NEAR,
    SIGNAL_TAKE,
    BASE_X,
    BASE_Y,
    Plan,
    PlanRejected,
    Subgoal,
    arena_problem,
    check_plan,
    signal_margin,
    home_plan,
    must_return,
    on_expensive_ground,
    parse_model_plan,
    plan_response_format,
    spendable_budget,
)
from did_llm.llm_client import LLMClient, LLMUnavailable
from did_llm.prompts import (
    build_planner_system,
    build_planner_prompt,
)


#: Events that stop the robot and force a fresh plan.
#:
#: These are the moments where the world stopped matching the plan: something
#: hurt, something was in the way, or a search came up empty. Continuing the
#: current plan after any of them means repeating the mistake, and the robot
#: has no way to know it happened without being told — the executor reports
#: it in /agent/status, and the planner is the thing that has to act on it.
STOP_EVENTS: frozenset[str] = frozenset({
    'collision',      # drove into something; the map is wrong or the route is
    'hazard_hit',     # paid a penalty for entering a dangerous zone
    'false_collect',  # reached for a sample that was not there
})

#: Anomaly flags that stop the robot, for the same reason. They come from
#: /agent/state rather than /did/events, so they are polled rather than
#: received.
STOP_ANOMALIES: tuple[str, ...] = (
    'penalties_burst',
    'sensor_noise_up',
)

#: Explains each stop to the model, so the feedback is a reason and not a word.
STOP_REASON = {
    'collision': 'произошло столкновение: карта или маршрут неверны, не повторяй этот путь',
    'hazard_hit': 'робот попал в опасную зону и заплатил штраф: обойди её',
    'false_collect': 'сбор не удался: образца там не было, поищи в другом месте',
    'penalties_burst': 'серия штрафов: тари�� зона опасна, уйди из неё',
    'sensor_noise_up': 'датчик шумит: одиночные показания больше ничего не значат,'
                       ' ориентируйся на статистику',
}


#: An empty ``subgoals`` list is the contract's stop signal, and publishing a
#: new plan preempts the running one. Together they are how the robot is held
#: still while a fresh plan is being written.
HOLD_PLAN = {'subgoals': []}


#: Search circle used when the reading says a sample is a step away. Tight,
#: because the point is to close the last half metre, not to sweep the area.
CLOSE_SEARCH_RADIUS_M = 0.35

#: Reach of the sample sensor in metres, so a reading can be turned back into a
#: distance. Taken from the scenario the judge runs.
SENSOR_RANGE_M = 1.5

#: Smallest difference in radius that counts as a wider circle. A hair wider is
#: not worth throwing away a search that is already running.
SEARCH_REACH_SLACK_M = 0.05

#: How long a «Стоп» is given to prove it was not part of a scenario restart.
#: The agent sends one on every scenario start and the new episode lands within a
#: second or two, so anything still unpaired after this is the operator.
STOP_GRACE_SEC = 8.0

#: While the reading places the sample inside this margin, a measured bearing is
#: worth stepping along. Below it the cone is too shallow for the direction to
#: mean much, and sweeping is the better use of the battery.
BEARING_STEP_MIN = 0.5
#: How far to step along the bearing. Short on purpose: each step is followed by
#: a fresh reading taken from closer in, so a bearing that was wrong by 30
#: degrees is corrected on the next one instead of compounding.
BEARING_STEP_MAX_M = 0.55

#: Below this margin the reading carries no usable information, and the plan is
#: sent somewhere unsearched instead of around the robot. Set above the noise
#: rather than at zero so a live sensor cannot trip it by twitching.
FAR_HUNT_MIN = 0.12
#: How far is far enough that it counts as leaving. Roughly the arena's own
#: scale, so the robot crosses ground it has not seen instead of re-walking the
#: cell it is standing next to.
FAR_HUNT_MIN_DIST_M = 2.0
#: Rough battery price of driving a metre, used only to refuse a crossing the
#: remaining charge cannot pay for.
FAR_HUNT_COST = 1.0
#: Radius for a search the planner itself inserts, taken where the reading gives
#: no width to work from.
SEARCH_RADIUS_DEFAULT = 0.8

#: The three signal readings the mission turns on, and the cases they name.
#: Below the low one the arena is simply unmeasured from here; above the middle
#: a sample is close and worth closing on; above the high one it is inside the
#: radius where collecting works at all.
SIGNAL_SKILL_LOW = 0.12
SIGNAL_SKILL_MID = 0.5
SIGNAL_SKILL_HIGH = 0.8
#: How far a reading has to move past a boundary before the case changes. A robot
#: circling a sample sits on the boundary and would otherwise re-fire every tick.
SKILL_HYSTERESIS = 0.05
#: Minimum gap between two skill handovers, so one crossing costs one call and
#: not one per tick while the reading settles.
SKILL_COOLDOWN_SEC = 20.0
#: How long the robot may stand still waiting for the model's answer before the
#: planner answers the case itself.
SKILL_WATCHDOG_SEC = 15.0

#: How many readings are kept, the fewest a plane can be fitted through, how far
#: apart they have to be for their differences to mean anything, and how strong
#: the best of them has to be before the cone is worth trusting. Below that
#: signal the cone around a sample is so flat that the fitted direction is
#: mostly noise, and driving along it would send the robot off on a guess.
TRACE_POINTS = 12
TRACE_MIN_VALID = 3
TRACE_MIN_SIGNAL = 0.25
#: How far the recovered position may be from each reading's own distance
#: before the readings are declared not to describe one sample.
#:
#: Measured, not guessed. Against the judge's sensor noise, accepting a looser
#: residual raises how often a bearing is produced but destroys how often it is
#: right: at 0.15 m it is right 67% of the time, at 0.35 m only 50%, at 0.7 m
#: 40%. So the strict setting, and what comes out is a hint for the model
#: rather than a heading the robot is sent along.
TRACE_MAX_RESIDUAL_M = 0.15
#: Closer than this and the robot is on the sample, so there is no bearing left
#: to give it.
TRACE_MIN_REACH_M = 0.12

#: How many swept circles are named in the prompt. Enough to cover an arena of
#: roughly this size, short enough that the block stays a paragraph rather than
#: the longest thing in it.
SEARCH_MEMORY = 12

#: Spacing of the candidate lattice handed to the model as places still worth
#: searching. Coarse on purpose: these are suggestions, and every one of them
#: costs a trip, so naming thirty would read as a queue to work through.
UNCOVERED_STEP = 1.2

#: How many candidate cells the prompt names. Enough that the model has a real
#: choice, few enough that it does not read as an instruction to tour them all.
UNCOVERED_HINTS = 5


def search_radius(margin: float) -> float:
    """The circle to sweep for a sample the reading puts ``margin`` above noise.

    A strong reading means a close sample, and a close sample is best closed in
    on with a tight circle: a wide one walks the robot back out of range it
    already has. A weak reading means the sample is somewhere in the neighbourhood
    and the circle has to be wide enough to sense it. The band between the two
    is where a circle narrower than the distance fails outright — it sweeps
    ground the sample is not on — so the circle is never allowed below what the
    reading implies.
    """
    if margin >= SIGNAL_CLOSE:
        base = CLOSE_SEARCH_RADIUS_M
    else:
        base = max(CLOSE_SEARCH_RADIUS_M, min(0.9, 1.6 * (1.0 - margin)))
    # Cover the distance the reading implies, within the slack.
    reaches = SENSOR_RANGE_M * (1.0 - margin) - SEARCH_REACH_SLACK_M
    return round(min(0.9, max(base, reaches)), 2)


def _nearest_solid(x: float, y: float) -> float:
    """How much room a point leaves: the gap to the nearest pillar or wall.

    Scored on the same numbers :func:`arena_problem` refuses a plan for, so a
    point it calls clear scores well and a point just past the limit scores near
    zero. Used to choose a way out of a stuck spot, where the point being
    technically legal matters less than the point actually being drivable.
    """
    arena = ARENA
    room = min(
        x - arena.x_min, arena.x_max - x,
        y - arena.y_min, arena.y_max - y,
    )
    for px, py, radius in arena.pillars:
        room = min(room, hypot(x - px, y - py) - max(PILLAR_KEEPOUT, radius + 0.2))
    return room


def _with_collect(subgoals: list[Subgoal]) -> list[Subgoal]:
    """Insert a ``collect`` after each ``search_around`` that lacks one."""
    result: list[Subgoal] = []
    for index, subgoal in enumerate(subgoals):
        result.append(subgoal)
        if subgoal.type != 'search_around':
            continue
        following = subgoals[index + 1].type if index + 1 < len(subgoals) else ''
        if following == 'collect':
            continue
        result.append(Subgoal(type='collect'))
    return result


class PlannerConfig:
    """Knobs the node exposes as ROS parameters."""

    def __init__(self, mission: str = 'Собрать образцы и вернуться на базу',
                 min_subgoals: int = 3,
                 max_subgoals: int = 6,
                 replan_period_sec: float = 30.0,
                 max_state_staleness_sec: float = 5.0,
                 repair_attempts: int = 1,
                 handover_after_failures: int = 3,
                 resume_after_sec: float = 20.0,
                 autonomous_fallback: bool = True,
                 interrupt_pause_sec: float = 12.0,
                 collision_pause_sec: float = 15.0,
                 collision_avoid_sec: float = 45.0,
                 collision_radius_m: float = COLLISION_RADIUS_M) -> None:
        self.mission = mission
        self.min_subgoals = min_subgoals
        self.max_subgoals = max_subgoals
        self.replan_period_sec = replan_period_sec
        self.max_state_staleness_sec = max_state_staleness_sec
        self.repair_attempts = repair_attempts
        self.handover_after_failures = handover_after_failures
        self.resume_after_sec = resume_after_sec
        #: Whether the episode may be handed to the agent's own behaviour when
        #: the model stops answering. Off for a run whose whole point is to
        #: measure the planner: a silent switch would attribute the agent's
        #: result to the model.
        self.autonomous_fallback = autonomous_fallback
        self.interrupt_pause_sec = interrupt_pause_sec
        self.collision_pause_sec = collision_pause_sec
        self.collision_avoid_sec = collision_avoid_sec
        self.collision_radius_m = collision_radius_m


class Planner:
    """Decides when to plan and what to publish.

    Deliberately free of ROS. The transport is whatever object is handed in:
    the live :class:`~did_llm.agent_link.AgentLink` inside the node, or the
    simulator-backed link in ``integration.py``. Both satisfy the same small
    surface — state, status, publish_plan, publish_command, journal, now — so
    the planning loop that runs in the demo is byte for byte the one that runs
    in the comparison, and neither can drift from the other.
    """

    def __init__(self, link: Any, client: LLMClient,
                 config: PlannerConfig | None = None) -> None:
        self.link = link
        self.client = client
        self.cfg = config or PlannerConfig()
        self.system_prompt = build_planner_system(self.cfg.max_subgoals)
        self.response_format = plan_response_format(self.cfg.max_subgoals)

        self.plan_counter = 0
        self.inflight: str | None = None
        self.inflight_since: float | None = None
        self.last_plan_at: float | None = None
        #: What was sent, so a running subgoal can be identified later:
        #: (plan_id, index, subgoal).
        self.sent_subgoals: list[tuple[str, int, Subgoal]] = []
        self.feedback: str = ''
        self.auto_requested = False
        #: Set once the episode has been handed to the agent for good; after that
        #: the planner stays out of the way but keeps probing for the model.
        self.handed_over = False
        self.handed_over_at = 0.0
        #: Consecutive model failures. One is a hiccup; a run of them is an
        #: outage.
        self.failures = 0
        #: Set by a stop event: the next tick must ask the model whatever the
        #: period says.
        self.force_replan = False
        #: Anomaly flags currently raised, for edge detection.
        self.seen_anomalies: set[str] = set()
        #: When the signal last forced an interruption, to keep the robot from
        #: being sent in circles.
        self.last_interrupt_at = -1e9
        #: Where the robot last hit something, and when: (x, y, t).
        self.collisions: list[tuple[float, float, float]] = []
        #: When the last collision triggered a stop, to debounce.
        self.last_collision_stop = -1e9
        #: Search circles already swept, so the model is not sent back to them.
        self.searched: list[tuple[float, float, float]] = []
        #: A model call in flight, and its result once it lands.
        self.busy = False
        self.busy_since = 0.0
        self.pending: tuple[str, Any, str, dict[str, float]] | None = None
        #: Counts planning rounds. It rotates the strategy directive, which is
        #: what keeps the endpoint's prompt cache from replaying an old answer.
        self.round_number = 0
        #: Set while the agent is driving itself, so it clears again when
        #: autonomous mode ends.
        self.yielded = False
        #: The operator pressed «Стоп»: stay quiet until the next run.
        self.quiet = False
        #: A «Стоп» seen but not yet acted on, with the time it arrived. The
        #: agent sends one of these when a scenario is started, so the planner
        #: waits to see whether a new episode follows before believing it.
        self.stop_pending = False
        self.stop_at = 0.0
        #: The judge's clock, watched only to notice it jumping back to zero.
        self.last_sim_t = 0.0
        #: Explicit /agent/state episode generation observed through AgentLink.
        #: This is the primary restart signal because a safe Burger respawn
        #: does not rewind the global Gazebo clock.
        self.last_episode_generation = int(
            getattr(self.link, 'episode_generation', 0))
        #: (x, y, sensor) as the robot walks, the raw material for a gradient.
        self.trace: list[tuple[float, float, float]] = []
        #: Searches that finished, and searches that were cut short. Separate
        #: lists because only the first was actually swept.
        self.attempted: list[tuple[float, float, float]] = []
        #: Subgoal reports already accounted for, so a completed search is
        #: recorded once.
        self._noted_status: set[tuple[str, Any, str]] = set()
        #: The signal case we are in, and the one waiting to be put to the model.
        self._skill_band: str | None = None
        #: The band last written about in the log, so a blocked skill says which
        #: guard is holding it instead of going quiet.
        self._skill_announced: str | None = None
        self.pending_skill: tuple[str, str] | None = None
        self.last_skill_at = 0.0
        #: Whether the last plan we sent was "go home".
        self._last_was_return = False
        #: Every rejection the executor reported, kept for the next prompt.
        self.rejections: list[str] = []
        #: Recent plans and their outcomes, so the model does not repeat itself.
        self.plan_history: list[dict[str, Any]] = []

    # ------------------------------------------------------------- statistics
    def stats(self) -> dict[str, Any]:
        return {
            **self.client.stats(),
            'plans': self.plan_counter,
            'rejections': len(self.rejections),
        }

    # ------------------------------------------------------------------ cycle
    def tick(self) -> None:
        """One decision. Cheap and silent when there is nothing to do."""
        if self.handed_over:
            # The model failed repeatedly. Keep probing slowly: a network blip
            # must not cost the whole episode, and the agent can still finish
            # it on its own meanwhile.
            self._probe_after_model()
            return
        if self.link.state is None:
            return
        if self.link.state_age() > self.cfg.max_state_staleness_sec:
            # The agent stopped publishing, which means it is not running or
            # the episode is over. Planning on a stale pose would be a guess.
            return
        if self.link.command_pending:
            self.link.command_pending = False
            self._on_operator_command(self.link.command)
        self._note_completed_search()
        if self._handle_signal_skill():
            return
        if self._new_episode():
            self._reset_for_new_episode()
        if self.stop_pending and self.link.now() - self.stop_at >= STOP_GRACE_SEC:
            # The stop was not part of a scenario restart, so it was the
            # operator. Silence until the next run, as asked for.
            self.stop_pending = False
            self.quiet = True
            self.inflight = None
            self.link.log.info(
                'оператор нажал «Стоп» — бургер не вмешивается до нового прогона')
        if self.quiet:
            return
        if self.link.finished():
            return
        if self.link.in_autonomous_mode():
            # The operator or the agent's own fallback has taken over. Stay
            # quiet and let it run; taking control back mid-episode would
            # interrupt it. Not permanent: when autonomous mode ends, the
            # planner picks up again.
            if not self.yielded:
                self.yielded = True
                self.inflight = None
                self.link.log.info('агент в автономном режиме, планирование приостановлено')
            return
        self.yielded = False

        self._trace_reading()

        outcome = self._last_outcome()
        if outcome is not None:
            self._note_outcome(outcome)
            return

        if not self.client.cfg.configured:
            self._go_autonomous('API не настроен')
            return

        if self._handle_event():
            return

        if self._stop_on_anomaly():
            return

        if self._operator_is_driving():
            # Someone else has the robot. Publishing over them is what made the
            # dashboard's buttons look broken: "Стоп", "На базу" and clicks on
            # the map all arrive as a plan, the agent starts obeying, and thirty
            # seconds later this planner replaces it and turns the robot back
            # out into the arena.
            if not self.yielded:
                self.yielded = True
                self.inflight = None
                self.link.log.info(
                    f'чужой план {self.link.status.get("plan_id")} выполняется '
                    f'({self.link.status.get("subgoal")}), планирование '
                    f'приостановлено')
            return

        if self._collect_when_close():
            return

        if self._interrupt_for_sample():
            return

        if must_return(self.link.battery(), self.link.return_cost()):
            if self._last_was_return:
                # We already sent it home and the judge has not ended the
                # episode. Going home again is the same plan, so repeating it
                # only burns calls; the agent's own policy can finish what we
                # started.
                self._go_autonomous('возврат не завершил эпизод')
                return
            self._last_was_return = True
            self._publish(home_plan(self._next_id(),
                                    'батареи не хватает на обход по карте '
                                    'стоимостей'), source='budget')
            return
        self._last_was_return = False

        if self.busy or self.pending is not None:
            # Collect a finished answer, or keep waiting. This has to be
            # checked before the period: while a call is in flight _due() is
            # false, so gating on it first would leave a finished answer
            # sitting unclaimed and the robot waiting forever.
            self._plan_with_model()
            return

        if not self._due():
            return
        # Mark the moment before the call: with a 20-50 s generation the tick
        # must not be able to start a second call while the first is in
        # flight, and the period is in sim seconds.
        self.last_plan_at = self.link.now()
        self._plan_with_model()

    # ------------------------------------------------------------------ model
    def _plan_with_model(self) -> None:
        """Start a plan request in the background, or collect a finished one.

        The call takes 20-50 seconds and can hang for the whole timeout. Run
        inline it would block the single-threaded ROS executor, and during those
        seconds the robot could not be interrupted for a signal, an event or an
        anomaly — it would simply stand there. So it runs on a thread and the
        tick stays responsive.
        """
        if self.pending is not None:
            self._collect_result()
            return
        if self.busy:
            return

        budget = self._budget()
        prompt = build_planner_prompt(
            self.cfg.mission, self.link.state or {},
            self.link.status, self.feedback,
            expensive=self.link.expensive_ground(), budget=budget,
            searched=self.searched_circles(),
            attempted=self.attempted_circles(),
            uncovered=self.uncovered_cells(),
            bearing=self.bearing_hint(),
            bearing_step=self.bearing_step(),
            skill=self.pending_skill,
            round_number=self.round_number,
            plan_history=self.plan_history,
        )
        self.busy = True
        self.busy_since = self.link.now()
        self.round_number += 1

        def work() -> None:
            try:
                answer = self.client.complete_json(self.system_prompt, prompt,
                                                   tag='plan',
                                                   response_format=self.response_format)
                self.pending = ('plan', answer, prompt, budget)
            except LLMUnavailable as error:
                self.pending = ('error', error, prompt, budget)
            except Exception as error:  # noqa: BLE001 - a thread must not die
                self.pending = ('error', LLMUnavailable(str(error)), prompt, budget)

        threading.Thread(target=work, daemon=True).start()

    def _collect_result(self) -> None:
        kind, payload, prompt, budget = self.pending
        self.pending = None
        self.busy = False

        if kind == 'error':
            error = payload
            if error.reason == 'rate_limited':
                return
            self.failures += 1
            if self.failures < self.cfg.handover_after_failures:
                self.link.log.warn(
                    f'модель не ответила ({self.failures}/'
                    f'{self.cfg.handover_after_failures}): {error}')
                return
            self._go_autonomous(
                f'модель недоступна {self.failures} раз подряд: {error}')
            return
        self.failures = 0
        self._accept_answer(payload, prompt, budget)

    def _accept_answer(self, answer: Any, prompt: str,
                       budget: dict[str, float]) -> None:
        for attempt in range(self.cfg.repair_attempts + 1):
            try:
                plan = parse_model_plan(
                    answer, self._next_id(),
                    max_subgoals=self.cfg.max_subgoals,
                )
                # Whatever I can fix without asking the model is fixed first,
                # before validation. Running the checks on a plan I am about to
                # change reports problems that are no longer there, and every
                # one of them costs a repair round with the robot standing
                # still. Trimming afterwards meant the premature-return_to_base
                # cut — the rule I added for that — never got the chance: the
                # plan was already rejected by the time _trim saw it.
                self._trim(plan, budget)
                problems = check_plan(
                    plan, self.link.expensive_ground(),
                    min_battery=budget.get('search_budget'),
                    samples_remaining=self.link.remaining_samples(),
                    signal_high=self.link.signal(),
                    sensor_noise=self.link.noise(),
                    pose=self.link.pose(),
                    battery=budget.get('battery'),
                    hits=self.recent_hits(),
                    plan_history=self.plan_history,
                )
                if problems:
                    # Well-formed but unusable: a point on a pillar, the same
                    # spot twice, a search with no collect. The model can fix
                    # all of these in one go if it is told about all of them.
                    raise PlanRejected('; '.join(problems))
            except PlanRejected as reason:
                self.rejections.append(str(reason))
                # To the log as well as the journal: a rejected plan is the
                # one thing that needs watching during a live run, and the
                # journal only reaches the dashboard.
                self.link.log.warn(f'план отклонён: {reason}')
                self.link.journal(
                    'result',
                    f'План отклонён проверкой: {reason}',
                    status='rejected',
                    source='llm_planner',
                )
                if attempt >= self.cfg.repair_attempts:
                    # Out of repairs, but the endpoint is alive: the model
                    # answered, it just answered badly. Rather than wait for
                    # another answer — the robot may be standing in the way of
                    # its own goal — send a plan of our own and move.
                    self.feedback = f'Прошлый план отклонён: {reason}. Исправь.'
                    escape = self.escape_plan('сам выбраться из застревания')
                    if escape is not None:
                        self._publish(escape, source='escape')
                        return
                    self.last_plan_at = self.link.now()
                    return
                try:
                    answer = self.client.complete_json(
                        self.system_prompt,
                        f'{prompt}\n\nТвой прошлый ответ отклонён: {reason}\n'
                        'Исправь и верни полный JSON заново.',
                        tag='repair',
                        response_format=self.response_format,
                    )
                except LLMUnavailable as error:
                    if error.reason == 'rate_limited':
                        return
                    self.failures += 1
                    if self.failures >= self.cfg.handover_after_failures:
                        self._go_autonomous(
                            f'модель не ответила на исправление {self.failures} '
                            'раз подряд')
                    return
                self.failures = 0
                continue

            self._publish(plan, source='llm')
            return

    def _trim(self, plan: Plan, budget: dict[str, float]) -> None:
        """Make the plan executable, then cut it down to a workable size.

        Three things happen here, in this order.

        First, a ``collect`` is added after every ``search_around`` that is not
        already followed by one. Searching and then not collecting is always a
        mistake — ``search_around`` deliberately ends next to a sample — and
        models omit it often enough that leaving it out silently scores zero.

        Then a premature ``return_to_base`` is dropped.

        Then the list is shortened. Trimming keeps the tail, not the head: the
        head is where the plan commits to a direction, and a cut that removes
        it leaves a plan that starts halfway through.
        """
        plan.subgoals = _with_collect(plan.subgoals)
        self._clamp_radii(plan)

        # A trailing ``return_to_base`` while samples remain ends the episode at
        # whatever has been collected, so it is refused. Refusing the whole plan
        # for it is wasteful, though: the searches in front of it are exactly
        # what the robot should be doing, and cutting the last subgoal off
        # leaves those. It was the single most common rejection — the model
        # reaches for it almost every round — and each one cost a repair call
        # while the robot stood still.
        if (plan.subgoals and plan.subgoals[-1].type == 'return_to_base'
                and len(plan.subgoals) > 1
                and self.link.remaining_samples()):
            floor = budget.get('search_budget')
            if floor is None or floor > SEARCH_WORTH_IT:
                plan.subgoals = plan.subgoals[:-1]
                if plan.explanation:
                    plan.explanation += (
                        ' Домой отложил: образцы ещё остались.')

        limit = self.cfg.max_subgoals
        if len(plan.subgoals) > limit:
            plan.subgoals = plan.subgoals[-limit:]
        if self.pending_skill is None:
            # Only when the model was not asked. A skill handover means the model
            # has just been given this exact decision, and rewriting its answer
            # with a rule of this planner's own would make the handover pointless:
            # the "датчик молчит" substitution fired eleven times in ninety-six
            # seconds and all but crowded the model out of its own plan.
            self._search_here_first(plan)
            self._send_far_when_silent(plan)

    def _send_far_when_silent(self, plan: Plan) -> None:
        """Replace a local search with a crossing, when the reading is dead.

        The robot sat on the base searching a circle it had no reason to search:
        the reading was 0.008 against a noise floor of 0.006, and the model, left
        to its own devices, kept handing back ``goto(-2.2, -0.7)`` — the cell at
        the base, again, for the third episode running.

        A reading that low is not a missed sample. The sensor reads nothing past
        its range, so near zero means the arena is simply unmeasured from here and
        the only way to learn anything is to move to ground that has not been
        swept. So the first move is sent to an unsearched cell far away, and the
        rest of the model's plan follows it.
        """
        margin = signal_margin(self.link.signal(), self.link.noise())
        if margin >= FAR_HUNT_MIN:
            return
        pose = self.link.pose()
        if pose is None:
            return
        first_move = next((item for item in plan.subgoals
                           if item.type in ('goto', 'search_around')), None)
        if first_move is None:
            return
        if hypot(first_move.x - pose[0], first_move.y - pose[1]) >= FAR_HUNT_MIN_DIST_M:
            return  # the model already means to go somewhere else
        target = self.far_uncovered_cell(FAR_HUNT_MIN_DIST_M)
        if target is None:
            return  # nothing far enough, or nothing the battery can pay for
        # Drop the local search being replaced together with its own collect.
        # Keeping them sent the robot out to the far cell and then straight back
        # to the base circle, which is the loop this is meant to end.
        tail: list[Subgoal] = []
        dropped_collect = False
        for item in plan.subgoals[plan.subgoals.index(first_move) + 1:]:
            if not dropped_collect:
                if item.type == 'collect':
                    dropped_collect = True
                continue
            tail.append(item)
        plan.subgoals = [
            Subgoal(type='goto', x=target[0], y=target[1]),
            Subgoal(type='search_around', x=target[0], y=target[1],
                    radius=SEARCH_RADIUS_DEFAULT),
            Subgoal(type='collect'),
            *tail,
        ]
        if plan.explanation:
            plan.explanation += (' Датчик молчит, поэтому иду в неосмотренную '
                                 'часть арены, а не кругу рядом.')
        self.link.log.info(
            f'   датчик молчит (запас {margin:.2f}) — вместо поиска рядом '
            f'иду в неосмотренную точку ({target[0]:g}; {target[1]:g}), '
            f'{hypot(target[0] - pose[0], target[1] - pose[1]):.1f} м отсюда')

    def _search_here_first(self, plan: Plan) -> None:
        """Put a local search and collect in front of a plan that ignores one.

        When the reading says a sample is close and the plan opens by driving
        somewhere else, the check refuses it: «сигнал 0.51 — образец в
        пределах досягаемости, сначала search_around и collect на месте, а не
        поездка в (1.4; -0.7)». That single rule was every remaining rejection
        in a run — seven of seven — and each cost a repair round with the robot
        standing still.

        Nothing about it needs refusing. The model wants to explore from
        somewhere else, and the sensor says there is something at the robot's
        feet. Both are satisfiable: search here, collect, and then go do what
        the model wrote. The model's plan is kept intact, the sample is picked
        up, and the call is not wasted.
        """
        margin = signal_margin(self.link.signal(), self.link.noise())
        if margin < SIGNAL_NEAR:
            return
        pose = self.link.pose()
        if pose is None:
            return
        first_move = next((item for item in plan.subgoals
                           if item.type in ('goto', 'search_around')), None)
        if first_move is None or first_move.type != 'goto':
            return
        # A goto to where the robot already stands means "here, now", which is
        # what a leading search says anyway.
        if hypot(first_move.x - pose[0], first_move.y - pose[1]) < GOTO_NOOP_M:
            return

        if margin >= SIGNAL_TAKE:
            why = 'сигнал выше 0.80 — образец рядом, беру сразу'
        else:
            why = 'образец рядом — сначала ищу здесь'
        # Always the pair, even when the sample is already in collect range: a
        # bare ``collect`` in front still leaves the model's ``goto`` as the
        # first movement, and the check refuses that just the same. The circle
        # is 0.35 m at this reading, so it does not walk the robot away from a
        # sample it can already take.
        head = [Subgoal(type='search_around',
                        x=round(pose[0], 2), y=round(pose[1], 2),
                        radius=round(search_radius(margin), 2)),
                Subgoal(type='collect')]
        plan.subgoals = head + plan.subgoals
        if plan.explanation:
            plan.explanation += f' {why}.'
        self.link.log.info(
            f'   добавил в начало плана: {why} (сигнал {margin:.2f}); '
            f'далее по плану модели: {first_move.describe()}')

    # ---------------------------------------------------------------- outcome
    def _last_outcome(self) -> dict[str, Any] | None:
        if self.inflight is None:
            return None
        status = self.link.last_status_for(self.inflight)
        if status is None:
            return None
        return status

    def _trace_reading(self) -> None:
        """Remember where the robot was and what the sensor said there.

        The sensor gives a magnitude and no bearing: it reads ``1 - d/1.5`` for
        the distance ``d`` to the nearest sample, and every point on that cone
        reads the same. A direction therefore cannot be read off a single
        reading — but it can be recovered from several, because the robot has
        already been walking around taking them. This records them so the
        gradient can be fitted later, which costs no extra driving.

        Only positions far enough apart to tell anything are kept: two readings
        2 cm apart differ by noise, not by distance to the sample.
        """
        pose = self.link.pose()
        if pose is None:
            return
        signal = self.link.signal()
        if not signal:
            return
        x, y = round(pose[0], 2), round(pose[1], 2)
        if self.trace and hypot(x - self.trace[-1][0],
                                y - self.trace[-1][1]) < 0.25:
            return
        self.trace.append((x, y, signal))
        if len(self.trace) > TRACE_POINTS:
            del self.trace[:-TRACE_POINTS]

    def sample_direction(self) -> tuple[float, float, float] | None:
        """Where the sensor says the sample lies, or None when it cannot say.

        The reading is not a gradient, it is a distance: the sensor reports
        ``1 - d/1.5`` for the distance ``d`` to the nearest uncollected sample,
        so each reading says how far away the sample was, and three such
        distances from three different places pin its position down exactly.
        That is trilateration, and it is solved here in closed form.

        Two obvious alternatives were built and measured first, and both fail on
        the moment that matters most — the robot standing next to the sample it
        just drove past.

        A plane fitted through the readings cannot describe a peak, and the
        reading is a cone: it climbs and falls. Fitting "signal ≈ a·x + b·y"
        returns the slope of the *approach*, which from where the robot now
        stands is the direction away from the sample. Over 300 simulated
        approaches that was right 41 times and wrong 259.

        Weighting each step by the signal change over it has the same blind spot
        for the same reason, and additionally collapses on a straight line,
        where a least-squares plane is not determined: walking due north at a
        sample due north came back heading west.

        Distances have neither problem. A peak is just a distance that grew and
        shrank, and the fit is exact rather than a slope, so nothing depends on
        the shape of the path.
        """
        # Only readings that carry distance. A clipped reading is a zero, not a
        # distance of 1.5, and putting those in would place the sample at the
        # rim of the sensor's range on every straight approach. That also means
        # the count needed is three, not the whole trace: a robot two metres out
        # reads nothing at all.
        used = [point for point in self.trace
                if point[2] >= TRACE_MIN_SIGNAL]
        if len(used) < TRACE_MIN_VALID:
            return None
        anchor = used[-1]
        d_0 = SENSOR_RANGE_M * (1.0 - anchor[2])

        # |p_i - s|² - |p_0 - s|² = d_i² - d_0²  expands to a linear equation in
        # the unknown sample position. All of them are solved together rather
        # than picking two: the robot's readings carry noise, and averaging
        # every usable one beats trusting any single pair.
        saa = sab = sbb = sac = sbc = 0.0
        for x, y, signal in used[:-1]:
            d_i = SENSOR_RANGE_M * (1.0 - signal)
            a = 2.0 * (anchor[0] - x)
            b = 2.0 * (anchor[1] - y)
            c = (anchor[0] ** 2 + anchor[1] ** 2 - x ** 2 - y ** 2
                 + d_i ** 2 - d_0 ** 2)
            saa += a * a
            sab += a * b
            sbb += b * b
            sac += a * c
            sbc += b * c
        det = saa * sbb - sab * sab
        if abs(det) < 1e-6:
            # Every extra reading lies on a line through the anchor, which pins
            # down one coordinate but not the other.
            return None
        sx = (sac * sbb - sbc * sab) / det
        sy = (saa * sbc - sab * sac) / det

        # Check the answer against every reading. If it does not reproduce them,
        # then they were not all about the same sample — a collection between
        # readings moves the sensor onto a different one, and a noise spike
        # inflates one distance. Either way there is nothing to report.
        worst = 0.0
        for x, y, signal in used:
            claimed = SENSOR_RANGE_M * (1.0 - signal)
            worst = max(worst, abs(hypot(sx - x, sy - y) - claimed))
        if worst > TRACE_MAX_RESIDUAL_M:
            return None

        here = self.link.pose()
        if here is None:
            return None
        off = hypot(sx - here[0], sy - here[1])
        if off < TRACE_MIN_REACH_M or off > SENSOR_RANGE_M:
            # On top of the sample, or out of range: no bearing to give.
            return None
        return (sx - here[0]) / off, (sy - here[1]) / off, off

    def bearing_step(self) -> tuple[float, float, float] | None:
        """A point to try next, along the measured bearing, if one is clear.

        The estimate is right about two times in three, so on its own it is too
        weak to send a robot after. What makes it usable is that a wrong bearing
        is only wrong by degrees: stepping a short way along it keeps the robot
        near the sample either way, and the next reading is taken from closer in
        and therefore better. So the step is deliberately short and repeated,
        which turns an unreliable heading into a converging walk.

        Computed here rather than asked of the model on purpose. "Head within
        30 degrees of the rising signal" is not something to hand to something
        that reasons in coordinates; the point is worked out, checked against the
        pillars, and offered as a number.
        """
        found = self.sample_direction()
        if found is None:
            return None
        dx, dy, distance = found
        pose = self.link.pose()
        if pose is None:
            return None
        # Only worth a directed step while the reading places the sample inside
        # the range where the cone is steep enough to mean anything.
        if signal_margin(self.link.signal(), self.link.noise()) < BEARING_STEP_MIN:
            return None
        step = min(distance, BEARING_STEP_MAX_M)
        # Turn the beam rather than give up: a pillar between here and there is
        # a reason to try a neighbouring bearing, not to stand still.
        for turn in (0.0, 0.35, -0.35, 0.7, -0.7, 1.05, -1.05):
            angle = atan2(dy, dx) + turn
            ux, uy = cos(angle), sin(angle)
            point = (round(pose[0] + ux * step, 2), round(pose[1] + uy * step, 2))
            if arena_problem(*point) is None:
                return point[0], point[1], step
        return None

    def far_uncovered_cell(self, min_distance: float) -> tuple[float, float] | None:
        """An unsearched cell worth crossing the arena for.

        Used when the reading says there is nothing within reach. Searching the
        ground the robot already stands on then costs battery and buys nothing:
        the sensor falls off to nothing past its range, so a reading near zero is
        not a sample nearby that was missed, it is the absence of information
        about the whole arena. The way out is to go and look somewhere else.
        """
        pose = self.link.pose()
        if pose is None:
            return None
        expensive = self.link.expensive_ground()
        battery = self.link.battery()
        budget = max(0.0, (battery or 0.0) - self.link.return_cost()) if battery else None

        arena = ARENA
        best: tuple[float, tuple[float, float]] | None = None
        y = arena.y_min + UNCOVERED_STEP / 2
        while y < arena.y_max:
            x = arena.x_min + UNCOVERED_STEP / 2
            while x < arena.x_max:
                point = (round(x, 2), round(y, 2))
                x += UNCOVERED_STEP
                if (arena_problem(*point) is not None
                        or self._was_searched(point)
                        or on_expensive_ground(point[0], point[1], expensive)):
                    continue
                distance = hypot(point[0] - pose[0], point[1] - pose[1])
                if distance < min_distance:
                    continue
                # Never further than the battery can pay for and still get home.
                if budget is not None and distance * 2 > budget * FAR_HUNT_COST:
                    continue
                if best is None or distance > best[0]:
                    best = (distance, point)
            y += UNCOVERED_STEP
        return best[1] if best else None

    def _handle_signal_skill(self) -> bool:
        """Act on a crossing of one of the three signal boundaries.

        Returns True when it handled the tick, either by stopping to ask the
        model or by answering mechanically because the model cannot be asked.
        """
        if self.pending_skill is not None:
            # The robot has already been stopped for this skill. If the model
            # has not answered by now, stop waiting: an episode standing still
            # is the one failure mode nothing recovers from, so the mechanical
            # answer goes out rather than the robot idling.
            if (self.link.now() - self.last_skill_at > SKILL_WATCHDOG_SEC
                    and not self.busy and self.pending is None):
                skill, why = self.pending_skill
                self.link.log.warn(
                    f'модель не ответила на скил «{skill}» за '
                    f'{SKILL_WATCHDOG_SEC:.0f} с — отвечаю сам, робот стоял')
                self.pending_skill = None
                self._publish(escape_plan(self._next_id(),
                                          'скил без ответа модели'),
                              source='skill-watchdog')
            return False
        if not self.client.cfg.configured:
            return False
        if self.busy or self.pending is not None:
            margin = signal_margin(self.link.signal(), self.link.noise())
            band_hint, _ = self._signal_skill(margin)
            if band_hint is not None and band_hint != self._skill_announced:
                self._skill_announced = band_hint
                self.link.log.info(
                    f'скил «{band_hint}» ждёт вызова модели '
                    f'(busy={self.busy})')
            return False
        margin = signal_margin(self.link.signal(), self.link.noise())
        band, is_new = self._signal_skill(margin)
        if not is_new:
            if self._skill_announced is not None and band != self._skill_announced:
                self._skill_announced = band
                self.link.log.info(
                    f'скил не вызывается: полоса «{band}» уже была '
                    f'({self._skill_band}), сигнал {margin:.2f}')
            return False
        self._skill_announced = band
        wait = SKILL_COOLDOWN_SEC - (self.link.now() - self.last_skill_at)
        if wait > 0:
            # Not yet, and say so: a crossing that is merely waiting must look
            # different in the log from one that is being ignored, because the
            # two fail in the same silent way.
            self.link.log.info(
                f'скил «{band}» ждёт {wait:.0f} с (запас {margin:.2f})')
            return False
        self._skill_band = band

        why = (f'сигнал {self.link.signal():.2f} при шуме '
               f'{self.link.noise() or 0.0:.2f} (запас {margin:.2f})')
        if self._ask_model_for_skill(band, why):
            return True
        return self._interrupt_for_sample()

    def _signal_skill(self, margin: float) -> tuple[str | None, bool]:
        """Which of the three signal cases we are in, and whether it is new.

        The bands are the three readings the mission turns on: below 0.12 there
        is nothing in reach, above 0.5 a sample is close, above 0.8 it is inside
        the radius where collecting works. Between 0.12 and 0.5 the reading says
        nothing actionable, so no case applies and the plan is left alone.

        Hysteresis on each band, because a robot circling a sample sits right on
        a boundary and a plain comparison would fire the same case every tick.
        """
        previous = self._skill_band
        # Escalation first, so rising from "close" to "in hand" is never
        # swallowed by the band we were already in — that is the transition that
        # matters, the sample is collectable now and not a moment later. The
        # hysteresis only ever holds a band open while the reading sits near its
        # lower edge, which is where a robot circling a sample would otherwise
        # chatter. Firing is on the band changing, never on the reading moving.
        if margin >= SIGNAL_SKILL_HIGH:
            band = 'take'
        elif previous == 'take' and margin >= SIGNAL_SKILL_HIGH - SKILL_HYSTERESIS:
            band = 'take'
        elif margin >= SIGNAL_SKILL_MID:
            band = 'approach'
        elif previous == 'approach' and margin >= SIGNAL_SKILL_MID - SKILL_HYSTERESIS:
            band = 'approach'
        elif margin < SIGNAL_SKILL_LOW:
            band = 'far'
        elif previous == 'far' and margin < SIGNAL_SKILL_LOW + SKILL_HYSTERESIS:
            band = 'far'
        else:
            band = None
        return band, band is not None and band != previous

    def _ask_model_for_skill(self, skill: str, why: str) -> bool:
        """Stop the robot and put the case to the model as a named skill.

        The three signal cases are decisions, not arithmetic: whether to close
        the last half metre, walk to a spot that was a moment ago empty, or cross
        the arena for ground nobody has swept are all judgement calls about
        battery and time. So the robot stops and the model is told which case it
        is in, rather than this planner deciding on its own.

        The mechanical answers are kept for when the model cannot be asked — no
        key, a dead endpoint, a call already in flight — because a stopped robot
        in a live episode is worse than either option.
        """
        self.pending_skill = (skill, why)
        self.force_replan = True
        self.last_skill_at = self.link.now()
        self.inflight = None
        # Written as the wire object rather than through Plan. An empty plan is
        # the stop, and Plan will not hold one — it requires at least one
        # subgoal — so building it here raised a validation error that escaped
        # the tick and was swallowed with no trace. Every skill handover was
        # dying on this one line: nothing was ever stopped, nothing was ever
        # asked, and the log showed nothing at all.
        self.link.publish_plan({
            'plan_id': self._next_id(),
            'subgoals': [],
            'explanation': f'остановка: скил «{skill}»',
        })
        self.link.log.warn(
            f'стоп перед скилом «{skill}»: {why} — спрашиваю модель')
        # Ask now, in this same tick. Leaving it to the next one left a window
        # in which the robot was stopped and nobody was going to move it: the
        # agent took the empty plan, went idle, and the run simply stood there.
        self._plan_with_model()
        return True

    def bearing_hint(self) -> tuple[float, float, float] | None:
        """The estimated sample position to show the model, if one is trusted.

        The estimate goes to the model as a hint and never becomes a subgoal of
        its own. It is not accurate enough to steer by: measured against the
        judge's sensor noise it lands within 20 degrees about two times in
        three, which is worth telling a model and not worth obeying blindly.
        """
        found = self.sample_direction()
        if found is None:
            return None
        dx, dy, distance = found
        pose = self.link.pose()
        if pose is None:
            return None
        return pose[0] + dx * distance, pose[1] + dy * distance, distance

    def _on_operator_command(self, command: str | None) -> None:
        """React to the dashboard's Стоп / Автономно.

        «Стоп» is the one instruction here that has no reason to be
        second-guessed. It used to be: the stop cancelled the goal but left no
        plan behind, so thirty seconds later this planner published a fresh one
        and the robot drove off again — which is what "кнопка не работает" looked
        like from the browser. So «Стоп» means quiet until the next run starts,
        and it stays quiet even if «Автономно» is pressed afterwards: the
        operator stopped the robot last, so the robot stands and this planner
        waits. Starting a new run is what lifts it.

        «Автономно» is a toggle rather than a one-way door. Handing over used to
        be final for the whole episode — this planner stood down and had no way
        back, because ``in_autonomous_mode()`` held for as long as the agent's
        own policy ran. A demo needs the opposite: an accidental press should
        cost one click, not the run. The agent's autonomous loop returns when a
        plan preempts it, so publishing one is enough to take the wheel back.
        """
        if command == 'stop':
            # Not applied at once. The agent sends ``stop`` as part of starting
            # a scenario too, and treating that as the operator halting the run
            # silenced the planner for good whenever the restart then failed to
            # complete: the robot stood against a wall for six minutes taking a
            # collision every two seconds and scoring minus 332, because no plan
            # ever arrived. The two cases are told apart by what follows — a
            # scenario restart brings a new episode within a second or two.
            self.stop_at = self.link.now()
            self.stop_pending = True
            return
        if command != 'auto':
            return
        if not self.handed_over:
            # The operator handed the robot to the agent's own policy. Record
            # it as our own handover state too, or the button cannot be a
            # toggle: the second press found handed_over still False, logged
            # "waiting", and left the agent driving for the rest of the episode.
            self.handed_over = True
            self.auto_requested = True
            self.handed_over_at = self.link.now()
            self.inflight = None
            self.link.log.info(
                'оператор отдал автономный режим — бургер ждёт. '
                'Повторное нажатие вернёт управление планировщику.')
            return
        # Second press: hand the wheel back to this planner.
        self.handed_over = False
        self.auto_requested = False
        self.handed_over_at = 0.0
        # A fresh set of attempts, or the failure counter that caused the
        # handover would immediately hand it back again.
        self.failures = 0
        self.force_replan = True
        self.last_plan_at = 0.0
        self.link.journal('decision', 'Возврат управления планировщику',
                          'Оператор вернул руль LLM-планировщику.')
        self.link.log.info('оператор вернул управление бургеру — снова планю')

    def _new_episode(self) -> bool:
        """Whether the judge has started a different run.

        AgentLink watches the explicit ``/agent/state.episode_id`` value. It
        changes even when the dashboard restarts the same scenario without
        rewinding Gazebo's global clock. The clock fallback keeps fake links
        and older transports compatible.
        """
        generation = getattr(self.link, 'episode_generation', None)
        if isinstance(generation, int) and generation != self.last_episode_generation:
            self.last_episode_generation = generation
            self.last_sim_t = self.link.now()
            self.link.log.info('новый прогон — бургер снова планирует')
            return True

        now = self.link.now()
        if self.last_sim_t and now < self.last_sim_t - 1.0:
            self.last_sim_t = now
            self.link.log.info('новый прогон — бургер снова планирует')
            return True
        self.last_sim_t = max(self.last_sim_t, now)
        return False

    def _reset_for_new_episode(self) -> None:
        """Drop planning memory that belongs to the previous judge run."""
        self.quiet = False
        self.stop_pending = False
        self.inflight = None
        self.sent_subgoals.clear()
        self.feedback = ''
        self.handed_over = False
        self.auto_requested = False
        self.yielded = False
        self.failures = 0
        self.force_replan = True
        self.seen_anomalies.clear()
        self.collisions.clear()
        self.searched.clear()
        # Readings taken in the previous run describe that run's samples. Kept,
        # they would point the gradient at a sample that is no longer there.
        self.trace.clear()
        self.attempted.clear()
        self._noted_status.clear()
        self.rejections.clear()
        self._last_was_return = False
        self.last_plan_at = None
        # Plan history belongs to the previous episode. Kept, it would steer
        # the model away from ground that is now worth searching again.
        self.plan_history.clear()

    def _operator_is_driving(self) -> bool:
        """Whether a plan this planner did not write is running on the agent.

        The operator's dashboard controls — ``Стоп``, ``На базу``, ``Собрать
        здесь``, a click on the map — all arrive as a plan on ``/agent/plan``,
        not as a mode switch. So they are only visible as a status whose plan
        is not ours, and the only way to respect them is to notice that and
        stay quiet for as long as it runs.
        """
        status = self.link.status
        if not status:
            return False
        plan_id = str(status.get('plan_id') or '')
        if not plan_id:
            return False
        if plan_id in self.link.seen_plans:
            return False
        if plan_id == (self.inflight or ''):
            return False
        # A plan that is not doing anything is not somebody driving. The agent
        # publishes a status carrying the default id ``plan`` with no subgoal
        # and ``idle``, and standing down for that left the robot jammed against
        # an obstacle while the judge logged a collision every two seconds: this
        # planner politely sat out a stall it should have been resolving. Every
        # operator control that really needs respecting — ``На базу``, ``Собрать
        # здесь``, a click on the map — arrives with a real subgoal, so asking
        # for one costs nothing.
        if not str(status.get('subgoal') or '').strip():
            return False
        return True

    def _clamp_radii(self, plan: Plan) -> None:
        """Pull any too-wide search down to what the reading allows.

        A radius over the cap is not a plan that cannot be executed, only one
        that is worse than it needs to be — a wide circle sweeps ground the
        sample is not on and walks the robot back out of range it already had.
        Refusing the whole plan for it cost a model call and left the robot
        standing still, which is how a marginally-too-wide circle turned into
        a minute of nothing happening.
        """
        margin = signal_margin(self.link.signal(), self.link.noise())
        allowed = max(0.4, 1.6 * max(0.0, 1.0 - margin))
        for subgoal in plan.subgoals:
            if subgoal.type != 'search_around' or subgoal.radius <= allowed + 0.05:
                continue
            shrunk = round(allowed, 2)
            self.link.log.info(
                f'   поправил радиус {subgoal.describe()}: '
                f'{subgoal.radius:g} → {shrunk:g} м (сигнал {margin:.2f})')
            subgoal.radius = shrunk

    def _note_outcome(self, status: dict[str, Any]) -> None:
        """Record how the last plan ended and turn it into the next prompt."""
        self.inflight = None
        state = status.get('state')
        reason = str(status.get('reason') or '')

        self.link.journal(
            'result',
            f'План {status.get("plan_id", "")}: {state}',
            text=reason,
            status='confirmed' if state == 'done' else 'rejected',
            source='llm_planner',
        )

        # Update the plan history with the outcome.
        plan_id = str(status.get('plan_id') or '')
        for entry in reversed(self.plan_history):
            if entry.get('plan_id') == plan_id:
                entry['outcome'] = state or 'unknown'
                entry['reason'] = reason
                break

        if state == 'failed' and reason:
            # Their handbook asks for exactly this: the executor's reason goes
            # back into the prompt rather than being summarised away.
            self.feedback = f'{reason}. Не повторяй этот план.'
            self.rejections.append(reason)
        else:
            self.feedback = ''

        if state == 'done':
            # Deliberately not resetting the clock here. This used to hold the
            # planner off for a full period after every completed plan, on the
            # reasoning that the robot had been standing still anyway — which is
            # backwards. It had just finished working; idling for thirty seconds
            # afterwards is what made it stand still. The replan period already
            # measures from when the plan was sent, so a plan that ran longer than
            # the period is replaced at once, and only a plan that finished early
            # waits out what is left of its period.
            if self._last_was_return:
                # We got home and the judge has not ended the episode. Going
                # home again is the same plan; the agent's own policy is the
                # thing that can finish what we started.
                self._go_autonomous('возврат на базу не завершил эпизод')

    # ------------------------------------------------------------------ reactions
    def _handle_event(self) -> bool:
        """Act on a penalty event: stop on trouble, note the rest.

        A successful collection is news but not an emergency — it does not
        stop the robot, it just clears the way for the next plan.
        """
        collected_seen = False
        while True:
            event = self.link.take_event()
            if event is None:
                # Give /agent/state one publishing tick to replace the sensor
                # value that still belonged to the sample just collected.
                return collected_seen
            name = str(event.get('event') or '')
            if name == 'sample_collected':
                collected = event.get('collected')
                if isinstance(collected, (int, float)):
                    self.link.note_collected(int(collected))
                state = self.link.state or {}
                total = state.get('samples_total')
                progress = ''
                if isinstance(collected, (int, float)):
                    progress = f'Собрано образцов: {int(collected)}'
                    if isinstance(total, (int, float)):
                        progress += f'/{int(total)}'
                    progress += '. '
                self.link.journal(
                    'llm',
                    'Получено новое состояние, готовится следующий план',
                    progress + 'Датчик теперь указывает на ближайший '
                    'оставшийся образец.',
                    source='llm_planner',
                )
                self.feedback = ('образец собран, ищи следующий: '
                                 'сейчас датчик снова указывает на ближайший')
                self.force_replan = True
                collected_seen = True
                continue
            if name == 'collision':
                self._note_collision()
                continue
            if name in STOP_EVENTS:
                self.stop_and_replan(name)
                return True

    def _note_collision(self) -> bool:
        """Remember where the robot hit something, and stop once per burst.

        The judge emits a collision at most every two seconds, so a robot that
        is stuck produces a steady stream of them. Two things follow.

        Only the first of a burst is recorded. A robot pressed against a wall
        does not move, so every later event carries the same position, and
        recording them all grows the forbidden area without ever moving the
        robot out of it.

        Only the first triggers a stop and a new plan. Reacting to each one
        means a model call every two seconds and a plan aimed at the same
        obstacle.
        """
        if self.link.now() - self.last_collision_stop < self.cfg.collision_pause_sec:
            return False

        pose = self.link.pose()
        if pose is not None:
            self.collisions.append((pose[0], pose[1], self.link.now()))
            del self.collisions[:-8]

        self.last_collision_stop = self.link.now()
        self.feedback = STOP_REASON['collision']
        self.stop_and_replan('collision')
        return True

    def escape_plan(self, why: str) -> Plan | None:
        """A plan of our own to get the robot moving again.

        Two things leave the robot stranded: the model keeps proposing targets
        next to the place it just hit, or its answer breaks a rule and there is
        nothing to send. Waiting for another answer is what keeps it stuck,
        because it is standing in the way of its own goal.

        The way out must be a *different* place, not the one it is stuck in.
        Searching where it already stands does nothing when the problem is that
        it is standing somewhere impossible — pressed against a pillar, say —
        which is exactly how it ends up colliding every two seconds.

        Heading for the middle of the arena sounds safe and is not: the base can
        sit behind the pillar the robot is wedged against, so a fixed direction
        walks it back into the same obstacle. An earlier version did that and
        spent five consecutive escapes driving five centimetres, each one long
        enough to touch the pillar again. So the direction is chosen instead of
        assumed — the one that leaves the most room from every obstacle, which is
        the one that can actually be driven.

        The plan is ours, so it skips validation: it exists precisely because
        nothing the model produced was acceptable.
        """
        pose = self.link.pose()
        if pose is None:
            return None

        # Collect points that already failed in recent plans so the escape
        # does not send the robot back to a place it could not reach.
        failed_points: list[tuple[float, float]] = []
        for entry in self.plan_history[-5:]:
            if entry.get('outcome') == 'failed':
                for sg in entry.get('subgoals', []):
                    if sg.get('type') in ('goto', 'search_around'):
                        failed_points.append((float(sg['x']), float(sg['y'])))

        spot = self._clearest_way_out(pose, exclude=failed_points)
        if spot is None:
            # Nothing around the robot is clear. A pillar is not a wall: the
            # robot can push past it, and standing still cannot go anywhere at
            # all. Refusing to leave here is how a stuck robot burns the rest of
            # the episode, so the shortest hop out is taken regardless.
            spot = (pose[0] + 0.6, pose[1])
            self.link.log.warn('план-замена: вокруг всё занято, иду сквозь')

        self.link.log.warn(f'план-замена: {why} — иду в ({spot[0]:.2f}; {spot[1]:.2f})')
        return Plan(
            plan_id=self._next_id(),
            subgoals=[Subgoal(type='goto', x=round(spot[0], 2), y=round(spot[1], 2)),
                      Subgoal(type='search_around',
                              x=round(spot[0], 2), y=round(spot[1], 2),
                              radius=CLOSE_SEARCH_RADIUS_M),
                      Subgoal(type='collect')],
            explanation=why,
        )

    def _clearest_way_out(self, pose: tuple[float, float],
                          step: float = 0.9,
                          exclude: list[tuple[float, float]] | None = None
                          ) -> tuple[float, float] | None:
        """The reachable point around the robot with the most room around it.

        Every direction is tried and scored by how far it leaves the robot from
        anything solid, rather than by pointing at some fixed place. Ties go to
        the point nearest the middle, so two equivalent ways out do not send the
        robot to the same corner twice.

        ``exclude`` lists points that already failed in recent plans. They are
        skipped so the escape does not send the robot back to a place it could
        not reach.
        """
        best = None
        best_score = None
        for degrees in range(0, 360, 15):
            angle = radians(degrees)
            point = (pose[0] + step * cos(angle), pose[1] + step * sin(angle))
            if arena_problem(*point) is not None:
                continue
            if exclude and any(hypot(point[0] - ex, point[1] - ey) < 0.5
                               for ex, ey in exclude):
                continue
            clearance = _nearest_solid(*point)
            # Prefer room; break ties toward the middle of the arena.
            score = (round(clearance, 2),
                     -hypot(point[0] - BASE_X, point[1] - BASE_Y))
            if best_score is None or score > best_score:
                best, best_score = point, score
        return best

    def collisions_near(self, x: float, y: float) -> int:
        """How many recent impacts are near a point."""
        now = self.link.now()
        return sum(1 for cx, cy, when in self.collisions
                   if now - when < self.cfg.collision_avoid_sec
                   and hypot(x - cx, y - cy) < self.cfg.collision_radius_m)

    def recent_hits(self) -> list[tuple[float, float]]:
        """Impact points still worth avoiding."""
        now = self.link.now()
        return [(x, y) for x, y, when in self.collisions
                if now - when < self.cfg.collision_avoid_sec]

    def _stop_on_anomaly(self) -> bool:
        """Stop on a raised anomaly flag, once per flag.

        These arrive polled in /agent/state rather than as events, so they need
        their own edge detection: without it the flag would stop the robot on
        every tick and it would never move again.
        """
        anomaly = (self.link.state or {}).get('anomaly')
        if not isinstance(anomaly, dict):
            return False
        raised = [name for name in STOP_ANOMALIES if anomaly.get(name)]
        fresh = [name for name in raised if name not in self.seen_anomalies]
        self.seen_anomalies = set(raised)
        if not fresh or self.inflight is None:
            return False
        self.stop_and_replan(fresh[0])
        return True

    def stop_and_replan(self, trigger: str) -> None:
        """Hold the robot still and ask for a new plan on the next tick.

        Used for the events that mean the world stopped matching the plan. The
        stop is immediate and costs one message; the model call then happens on
        the next tick with the trigger in the prompt. Standing still for the
        half minute the call takes costs a little battery, which is the right
        price: driving on a plan that is known to be wrong costs far more.
        """
        self.inflight = None
        self.sent_subgoals = []
        self.link.publish_plan(dict(HOLD_PLAN))
        self.force_replan = True
        self.feedback = STOP_REASON.get(trigger, f'событие: {trigger}')
        self.link.journal('decision',
                          f'Стоп и перепланирование: {trigger}',
                          text=self.feedback, source='llm_planner')

    def _interrupt_for_sample(self) -> bool:
        """Cut the current plan short when a sample turns out to be close.

        A plan is checked against the signal at the moment it is made, but the
        signal changes while the plan runs: the robot drives a few metres and
        the reading climbs from 0.1 to 0.8 because a sample is right there. The
        plan then says to leave for a point it guessed, and the robot drives
        away from something collectible.

        No model call is made here. The sensor gives magnitude and no
        direction, so when the reading is high the only sensible move is to
        search where the robot already stands — there is nothing for a model to
        decide, and waiting half a minute for it to be told so would waste more
        battery than the interruption saves.
        """
        signal = self.link.signal()
        # Judged against the noise, not against it. A noisy reading is weaker
        # evidence, not useless evidence: the sensor fault in the hard scenario
        # runs for minutes, and treating every reading during it as void left
        # the robot driving past samples a few tens of centimetres away.
        margin = signal_margin(signal, self.link.noise())
        if margin < SIGNAL_NEAR:
            return False

        subgoal = self._current_subgoal()
        if subgoal is not None and subgoal.type not in ('goto', 'search_around'):
            return False
        # A subgoal of None means nothing of ours is running: the last plan has
        # ended and the robot is standing still. It used to mean "nothing to
        # interrupt", which left the robot idle for the whole 30 s replan
        # period while a sample sat 0.30 m away and the reading said 0.79. A
        # close sample is worth collecting whether or not a plan happens to be
        # running, and this needs no model call to decide it.
        idle = subgoal is None

        pose = self.link.pose()
        if pose is None:
            return False
        # A search in progress is normally left alone. Only two cases justify
        # replacing it, and both are about the circle failing rather than
        # merely differing from what this planner would have written.
        #
        # Too narrow: the circle sweeps ground the sample is not on. A 0.5 m
        # circle 0.65 m from the sample turned on the spot until its budget ran
        # out with the reading at 0.57 throughout.
        #
        # Too wide, but only once the sample is close: a wide circle then drives
        # the robot back out of range it already had. A 0.7 m circle while the
        # sample sat 0.26 m away and the reading said 0.80 — inside collect
        # range — circled out of reach and could not collect.
        #
        # The narrow test must not also fire on mere difference. Comparing both
        # ways unconditionally was a mistake: the model writes whatever radius
        # it likes, so every circle that was not exactly this planner's number
        # triggered a replacement and the robot was preempted every 13-30 s. It
        # never finished one search — the radii ran 0.9, 0.3, 0.6, 0.5, 0.35,
        # 0.9, 0.4 while the robot sat in one spot with the same subgoal showing
        # as "running".
        #
        # Comparing the running circle against a radius derived from the reading
        # does not work either, because that target moves with the reading. The
        # signal breathes between 0.32 and 0.60 while the robot circles a sample
        # it is closing on, the wanted radius swings between 0.45 and 0.9 with
        # it, and every swing is a reason to replace the circle that is about to
        # finish. The centres ran 1.01, 1.98, 1.85, 2.07, 1.93, 1.89 and the
        # sample stood still the whole time.
        #
        # So a search under way is not replaced for being the wrong size. The
        # reading still gets its two decisive sayings: take the sample when it is
        # within collecting distance, and go somewhere else when the circle is
        # nowhere near the robot.
        wanted = search_radius(margin)
        if (not idle and subgoal.type == 'search_around'
                and hypot(subgoal.x - pose[0], subgoal.y - pose[1]) < 0.6
                and margin < SIGNAL_TAKE):
            return False
        # A live signal persists while the robot circles the sample, so without
        # a pause the interrupt would fire every tick and the robot would spin
        # in place instead of searching once and collecting.
        if self.link.now() - self.last_interrupt_at < self.cfg.interrupt_pause_sec:
            return False

        # Above 0.80 the sample is already within the radius where collect
        # succeeds, so searching would drive the robot around it and out of that
        # radius again — the reading that says "take it" is the one moment not to
        # look for it. Their search calls 0.7 found, which is 0.45 m and still too
        # far, so the spiral would walk away from a sample the robot could have
        # taken standing still.
        if margin >= SIGNAL_TAKE:
            subgoals = [Subgoal(type='collect')]
            why = 'сигнал выше 0.80 — образец в пределах сбора, беру сразу'
        else:
            radius = wanted
            subgoals = [Subgoal(type='search_around',
                                x=round(pose[0], 2), y=round(pose[1], 2),
                                radius=round(radius, 2)),
                        Subgoal(type='collect')]
            why = 'образец рядом, ищу на месте'

        plan = Plan(
            plan_id=self._next_id(),
            subgoals=subgoals,
            explanation=(f'сигнал датчика {signal:.2f} при шуме '
                         f'{(self.link.noise() or 0.0):.2f}: {why}'),
        )
        self.link.log.warn(
            f'сигнал {signal:.2f} шум {(self.link.noise() or 0.0):.2f} '
            f'(запас {margin:.2f}), текущая подцель '
            f'{subgoal.describe() if subgoal else "робот стоит"} — '
            'прерываю план, образец рядом')
        self.last_interrupt_at = self.link.now()
        self._publish(plan, source='signal')
        # Not forcing a replan here. The model used to be told to take over
        # immediately, and it did: this planner published "search here, then
        # collect" and the model answered with a six-subgoal plan of its own
        # within zero seconds, preempting it. The two-subgoal plan never ran, so
        # the robot kept changing course mid-search — the interrupt fired on a
        # 0.49 reading, the model answered, eleven seconds later another
        # interrupt, and no search was ever allowed to finish. The next plan is
        # due when this one ends, which is what _due() already arranges.
        return True

    def _collect_when_close(self) -> bool:
        """Collect immediately when the noise-adjusted signal reaches 0.80.

        The public sensor model is ``1 - distance / 1.5``, so a margin of 0.80
        means the sample is within the judge's 0.30 m collection radius.
        Starting another search here can only drive the robot away from an
        already collectible sample. The judge still performs the authoritative
        distance check when the ``collect`` subgoal executes.
        """
        signal = self.link.signal()
        noise = self.link.noise()
        margin = signal_margin(signal, noise)
        if signal is None or margin < SIGNAL_TAKE:
            return False

        current = self._current_subgoal()
        if self.inflight is not None:
            # Do not replace a plan before its first status arrives, and do not
            # interrupt a collection that is already in progress.
            if current is None or current.type == 'collect':
                return False

        if self.link.now() - self.last_interrupt_at < self.cfg.interrupt_pause_sec:
            return False

        plan = Plan(
            plan_id=self._next_id(),
            subgoals=[Subgoal(type='collect')],
            explanation=(f'сигнал {signal:.2f}, шум {(noise or 0.0):.2f}: '
                         'образец уже в зоне 0.30 м, собираю автоматически'),
        )
        self.link.log.info(
            f'сигнал {signal:.2f}, шум {(noise or 0.0):.2f}, '
            f'запас {margin:.2f} >= {SIGNAL_TAKE:.2f}: автоматический сбор'
        )
        self.last_interrupt_at = self.link.now()
        self._publish(plan, source='auto_collect')
        self.force_replan = True
        return True

    def _remember_search(self, subgoals: list[Subgoal]) -> None:
        """Note ground that a search has actually finished sweeping.

        The sample sensor says how close the nearest sample is and never which
        way, and no map in the prompt shows where samples are, so the model picks
        targets blind. The one thing that stops it picking the same blind target
        twice is being told what it already covered — and that record has to
        live here, because the model has no memory between rounds and the pose
        history does not distinguish "searched and empty" from "never went
        there".

        Only completed searches are recorded. Marking them when the plan was
        published meant a circle the robot never reached — the plan was
        preempted first — was still reported to the model as "searched, empty",
        so the prompt steered it away from ground nobody had looked at.
        """
        for subgoal in subgoals:
            if subgoal.type != 'search_around':
                continue
            circle = (subgoal.x, subgoal.y, subgoal.radius)
            if any(hypot(circle[0] - x, circle[1] - y) < 0.05
                   for x, y, _ in self.searched):
                continue
            self.searched.append(circle)
        del self.searched[:-SEARCH_MEMORY]

    def _remember_attempt(self, subgoals: list[Subgoal]) -> None:
        """Note a search that was cut short, kept apart from the swept ones.

        The circle was not covered, so it must not be reported as covered. But
        the robot stood there and left with nothing, and that is exactly what the
        model needs to stop choosing it again.
        """
        for subgoal in subgoals:
            if subgoal.type != 'search_around':
                continue
            circle = (subgoal.x, subgoal.y, subgoal.radius)
            if any(hypot(circle[0] - x, circle[1] - y) < 0.05
                   for x, y, _ in self.attempted):
                continue
            self.attempted.append(circle)
        del self.attempted[:-SEARCH_MEMORY]

    def _note_completed_search(self) -> None:
        """Record a search the agent has just reported as done.

        The agent publishes one status per subgoal, so this is the only place
        that knows a circle was really swept. Reading it off the terminal status
        of the subgoal in flight keeps the coverage record honest: it says what
        happened rather than what was intended.
        """
        status = self.link.status
        if not status:
            return
        marker = (str(status.get('plan_id') or ''), status.get('index'),
                  str(status.get('state') or ''))
        if marker in self._noted_status or marker[2] != 'done':
            return
        if marker[0] != (self.inflight or '') or marker[1] is None:
            return
        self._noted_status.add(marker)
        if len(self._noted_status) > 512:
            self._noted_status.clear()
        try:
            index = int(marker[1])
        except (TypeError, ValueError):
            return
        state = str(status.get('state') or '')
        for plan_id, position, subgoal in self.sent_subgoals:
            if plan_id != marker[0] or position != index:
                continue
            if subgoal.type != 'search_around':
                return
            if state == 'done':
                self._remember_search([subgoal])
            elif state in ('preempted', 'failed'):
                self._remember_attempt([subgoal])
            return

    def searched_circles(self) -> list[tuple[float, float, float]]:
        """The circles the model is told have already been swept."""
        return list(self.searched)

    def attempted_circles(self) -> list[tuple[float, float, float]]:
        """Circles a search was cut short in, so the model stops choosing them."""
        return list(self.attempted)

    def uncovered_cells(self, limit: int = UNCOVERED_HINTS
                        ) -> list[tuple[float, float]]:
        """Free arena cells that no search has swept yet, nearest first.

        The model picks its own targets, but it picks them blind: the sensor says
        how far the nearest sample is and never which way, and no map shows where
        samples are. Telling it only where it has been leaves it inventing a
        replacement, which is the same blind guess under a new name. This gives
        it somewhere real to choose from — the coverage it is missing, sorted so
        the cheap nearby ones come first and the battery pays for distance only
        when there is nothing closer left.
        """
        pose = self.link.pose() or (BASE_X, BASE_Y)
        arena = ARENA
        expensive = self.link.expensive_ground()
        # Under the «искать дальше» skill the nearest cells are the wrong
        # answer: the reading says there is nothing here, and the nearest
        # unsearched ground is the ground the robot is already standing on. The
        # two halves of the prompt were contradicting each other — the skill
        # said "cross the arena" while the target list handed over the cell at
        # the base — and the model believed the list.
        go_far = self.pending_skill is not None and self.pending_skill[0] == 'far'
        cells = []
        y = arena.y_min + UNCOVERED_STEP / 2
        while y < arena.y_max:
            x = arena.x_min + UNCOVERED_STEP / 2
            while x < arena.x_max:
                point = (round(x, 2), round(y, 2))
                if (arena_problem(*point) is None
                        and not self._was_searched(point)
                        and not on_expensive_ground(point[0], point[1],
                                                    expensive)):
                    cells.append((hypot(point[0] - pose[0], point[1] - pose[1]),
                                  point))
                x += UNCOVERED_STEP
            y += UNCOVERED_STEP
        if go_far:
            cells.sort(key=lambda item: -item[0])
        else:
            cells.sort(key=lambda item: item[0])
        return [point for _, point in cells[:limit]]

    def _was_searched(self, point: tuple[float, float]) -> bool:
        """Whether this spot has been dealt with, one way or another.

        An interrupted search counts. Its circle was never swept, so calling it
        "searched, empty" in the prompt would be a lie — which is why they are
        kept apart — but the robot did go there and did come away with
        nothing, so sending the model back is how it looped. The model was
        repeatedly handed (-2.2; 1.7) again, having no way to know the robot had
        already tried and been pulled away.
        """
        return any(hypot(point[0] - x, point[1] - y) <= r
                   for x, y, r in self.searched + self.attempted)

    def _current_subgoal(self) -> Subgoal | None:
        """The subgoal the executor is on, taken from the plan we sent."""
        status = self.link.status
        if not status or status.get('plan_id') != self.inflight:
            return None
        index = int(status.get('index') or 0)
        for sent in self.sent_subgoals:
            if sent[0] == self.inflight and sent[1] == index:
                return sent[2]
        return None

    def _budget(self) -> dict[str, float]:
        """What may be spent on new work after reserving the trip home."""
        battery = self.link.battery()
        return_cost = self.link.return_cost()
        spendable = spendable_budget(battery, return_cost)
        return {
            'search_budget': round(spendable, 1),
            'cost_to_come_back': round(max(0.0, return_cost or 0.0), 2),
            'battery': round(battery, 1),
        }

    def _probe_after_model(self) -> None:
        """Try to take planning back after the model failed repeatedly.

        The endpoint at this hackathon drops connections for tens of seconds at
        a time, so a single bad patch costs half an episode if the handover is
        permanent. Probing costs one request every ``resume_after_sec`` and the
        agent keeps working meanwhile, so there is no reason to stay away.
        """
        if self.link.now() - self.handed_over_at < self.cfg.resume_after_sec:
            return
        if self.link.in_autonomous_mode():
            # The operator or the agent chose this. Taking control back would
            # override a human decision.
            return
        prompt = build_planner_prompt(
            self.cfg.mission, self.link.state or {},
            self.link.status, 'сеть отвечает снова, продолжим планирование',
            plan_history=self.plan_history,
        )
        try:
            self.client.complete_json(
                self.system_prompt, prompt, tag='probe',
                response_format=self.response_format,
            )
        except LLMUnavailable as error:
            if error.reason == 'rate_limited':
                return
            self.handed_over_at = self.link.now()
            return
        self.handed_over = False
        self.failures = 0
        self.last_plan_at = 0.0
        self.link.log.info('модель снова отвечает, планирование возобновлено')
        self.link.journal('decision', 'Планирование возобновлено после сбоя сети')

    def _due(self) -> bool:
        if self.force_replan:
            return True
        if self.inflight is not None:
            return False
        if self.last_plan_at is None:
            return True
        if not self._search_in_progress():
            return self.link.now() - self.last_plan_at >= self.cfg.replan_period_sec
        # A search that is already under way is left to finish. It was not: the
        # period expired mid-circle, a fresh plan arrived, and every search in
        # every run ended the same way — "preempted: replaced by a new plan".
        # Nothing was ever covered, the robot drove between the same two cells
        # for minutes, and it collected one sample a minute and then stood in
        # circles near the last one. Only the sensor has a reason to cut a
        # search short, and _interrupt_for_sample() is that reason.
        return False

    def _search_in_progress(self) -> bool:
        """Whether the agent is running a search right now.

        The state matters as much as the type. A status left at "preempted"
        still names its subgoal, and reading the type alone would count that as
        a search in progress for the rest of the episode — the planner would
        wait for a circle that was already cancelled and never send another.
        """
        status = self.link.status or {}
        if str(status.get('state') or '') != 'running':
            return False
        subgoal = self._current_subgoal()
        return subgoal is not None and subgoal.type == 'search_around'

    def _go_autonomous(self, why: str = 'модель недоступна') -> None:
        """Hand over to the agent's own behaviour, once.

        The agent can finish the episode unaided, so a dead API costs the plan
        quality and not the run.
        """
        if self.handed_over:
            return
        if not self.cfg.autonomous_fallback:
            # LLM-only run. Waiting is the honest behaviour: handing over would
            # let the agent's own policy finish the episode, and the result
            # would then say nothing about the planner.
            self.handed_over_at = self.link.now()
            self.link.log.warn(
                f'модель недоступна ({why}); автономный режим запрещён, жду')
            return
        self.auto_requested = True
        self.handed_over = True
        self.handed_over_at = self.link.now()
        self.inflight = None
        self.link.publish_command('auto')
        self.link.journal('decision', f'Переход в автономный режим: {why}')
        self.link.log.warn(f'передача в автономный режим: {why}')

    def _publish(self, plan: Plan, *, source: str) -> None:
        self.pending_skill = None
        # Read the clock before updating ``last_plan_at`` so the log reports
        # the real interval. The first locally generated plan has no previous
        # timestamp and therefore reports zero seconds.
        now = self.link.now()
        since = 0.0 if self.last_plan_at is None else now - self.last_plan_at
        payload = plan.to_wire()
        payload['source'] = source
        self.link.publish_plan(payload)
        self.inflight = plan.plan_id
        self.last_plan_at = now
        self.force_replan = False
        # Coverage is not recorded here. A circle only counts once the agent
        # reports its search done, which _note_completed_search() watches for;
        # recording it on publication would claim ground the robot never reached.
        self.sent_subgoals = [(plan.plan_id, index, subgoal)
                              for index, subgoal in enumerate(plan.subgoals)]
        # Any plan that ends at the base is a decision to stop exploring, so
        # the next one that would do the same thing is the loop to guard
        # against, whether the model wrote it or the budget forced it.
        self._last_was_return = bool(
            plan.subgoals and plan.subgoals[-1].type == 'return_to_base')
        # Record the plan in history so the model can see what was already tried.
        self._record_plan(plan, source)
        if not plan.subgoals:
            self.link.log.warn(
                f'БУРГЕР → АГЕНТ   план {plan.plan_id} ({source}) ПУСТОЙ '
                f'— это команда «стоп», эпизод встанет')
            return
        body = '\n'.join(
            f'   {index + 1}. {item.describe()}'
            for index, item in enumerate(plan.subgoals))
        self.link.log.info(
            '\n'.join([
                '',
                '─' * 62,
                f'БУРГЕР → АГЕНТ   план {plan.plan_id}   '
                f'источник: {source}   {len(plan.subgoals)} подц.   '
                f'прошло с прошлого плана {since:.0f} с',
                '─' * 62,
                body,
            ]))
        if plan.explanation:
            self.link.log.info(f'   почему: {plan.explanation}')
        # No journal entry here on purpose: the agent's dashboard already turns
        # a plan's "explanation" into a `decision` entry, so writing one too
        # would show every plan twice in the feed.

    def _record_plan(self, plan: Plan, source: str) -> None:
        """Record a plan in history so the model can see what was already tried.

        The model has no memory between rounds. Without a record of what was
        already tried, it picks the same targets again: a plan that failed with
        "no path to goal" is followed by an identical plan, and a search that
        found nothing is repeated at the same coordinates. The history is the
        only thing that stops it.

        Each entry carries the plan's subgoals and the source (llm, budget,
        signal, etc.), so the model can see not just what was tried but why.
        The outcome is added later by _note_outcome when the plan finishes.
        """
        self.plan_history.append({
            'plan_id': plan.plan_id,
            'subgoals': [
                {
                    'type': sg.type,
                    'x': round(sg.x, 2),
                    'y': round(sg.y, 2),
                    'radius': round(sg.radius, 2),
                }
                for sg in plan.subgoals
            ],
            'source': source,
            'outcome': 'pending',
            'reason': '',
        })
        # Keep only the last 10 plans to avoid unbounded growth.
        del self.plan_history[:-10]

    def _next_id(self) -> str:
        self.plan_counter += 1
        return f'llm-{self.plan_counter:03d}'

    def summary(self) -> str:
        return json.dumps(self.stats(), ensure_ascii=False)
