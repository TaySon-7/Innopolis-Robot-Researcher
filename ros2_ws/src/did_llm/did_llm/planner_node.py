"""The planning loop: state in, plan out.

The order of the checks matters more than the model call. Before asking
anything, the loop settles what can be decided without a model: the episode is
over, the budget forbids going out, or the API is down. Each of those has a
deterministic answer, and spending a model call to rediscover it would make
the planner both slower and less reliable.
"""

from __future__ import annotations

import json
from math import hypot
import threading
from typing import Any

from did_llm.agent_plan import (
    NOISE_UNTRUSTWORTHY,
    SIGNAL_NEAR,
    Plan,
    PlanRejected,
    Subgoal,
    check_plan,
    home_plan,
    must_return,
    parse_model_plan,
    spendable_budget,
)
from did_llm.llm_client import LLMClient, LLMUnavailable
from did_llm.prompts import (
    PLANNER_SYSTEM,
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
                 interrupt_pause_sec: float = 12.0) -> None:
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
        #: Whether the last plan we sent was "go home".
        self._last_was_return = False
        #: Every rejection the executor reported, kept for the next prompt.
        self.rejections: list[str] = []

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

        outcome = self._last_outcome()
        if outcome is not None:
            self._note_outcome(outcome)
            return

        if not self.client.cfg.configured:
            self._go_autonomous('API не настроен')
            return

        if self._interrupt_for_sample():
            return

        if self._handle_event():
            return

        if self._stop_on_anomaly():
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
            round_number=self.round_number,
        )
        self.busy = True
        self.busy_since = self.link.now()
        self.round_number += 1

        def work() -> None:
            try:
                answer = self.client.complete_json(PLANNER_SYSTEM, prompt,
                                                   tag='plan')
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
                plan = parse_model_plan(answer, self._next_id())
                problems = check_plan(
                    plan, self.link.expensive_ground(),
                    min_battery=budget.get('floor'),
                    samples_remaining=self.link.remaining_samples(),
                    signal_high=self.link.signal(),
                    sensor_noise=self.link.noise(),
                    pose=self.link.pose(),
                    battery=budget.get('battery'),
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
                    # answered, it just answered badly. That is what the
                    # feedback channel is for, so the reason goes into the
                    # next prompt instead of handing the episode away.
                    self.feedback = f'Прошлый план отклонён: {reason}. Исправь.'
                    self.last_plan_at = self.link.now()
                    return
                try:
                    answer = self.client.complete_json(
                        PLANNER_SYSTEM,
                        f'{prompt}\n\nТвой прошлый ответ отклонён: {reason}\n'
                        'Исправь и верни полный JSON заново.',
                        tag='repair',
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

            self._trim(plan)
            self._publish(plan, source='llm')
            return

    def _trim(self, plan: Plan) -> None:
        """Make the plan executable, then cut it down to a workable size.

        Two things happen here, in this order.

        First, a ``collect`` is added after every ``search_around`` that is not
        already followed by one. Searching and then not collecting is always a
        mistake — ``search_around`` deliberately ends next to a sample — and
        models omit it often enough that leaving it out silently scores zero.

        Then the list is shortened. Trimming keeps the tail, not the head: the
        head is where the plan commits to a direction, and a cut that removes
        it leaves a plan that starts halfway through.
        """
        plan.subgoals = _with_collect(plan.subgoals)

        limit = self.cfg.max_subgoals
        if len(plan.subgoals) > limit:
            plan.subgoals = plan.subgoals[-limit:]

    # ---------------------------------------------------------------- outcome
    def _last_outcome(self) -> dict[str, Any] | None:
        if self.inflight is None:
            return None
        status = self.link.last_status_for(self.inflight)
        if status is None:
            return None
        return status

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

        if state == 'failed' and reason:
            # Their handbook asks for exactly this: the executor's reason goes
            # back into the prompt rather than being summarised away.
            self.feedback = f'{reason}. Не повторяй этот план.'
            self.rejections.append(reason)
        else:
            self.feedback = ''

        if state == 'done':
            # A finished plan means the robot stood still for the whole
            # segment; going out again immediately would waste a call.
            self.last_plan_at = self.link.now()
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
        while True:
            event = self.link.take_event()
            if event is None:
                return False
            name = str(event.get('event') or '')
            if name == 'sample_collected':
                collected = event.get('collected')
                if isinstance(collected, (int, float)):
                    self.link.note_collected(int(collected))
                self.feedback = ('образец собран, ищи следующий: '
                                 'сейчас датчик снова указывает на ближайший')
                self.force_replan = True
                continue
            if name in STOP_EVENTS:
                self.stop_and_replan(name)
                return True

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
        if signal is None or signal < SIGNAL_NEAR:
            return False
        noise = self.link.noise()
        if noise is not None and noise >= NOISE_UNTRUSTWORTHY:
            return False

        subgoal = self._current_subgoal()
        if subgoal is None or subgoal.type not in ('goto', 'search_around'):
            return False

        pose = self.link.pose()
        if pose is None:
            return False
        # Already searching where it stands: let it finish.
        if (subgoal.type == 'search_around'
                and hypot(subgoal.x - pose[0], subgoal.y - pose[1]) < 0.6):
            return False
        # A live signal persists while the robot circles the sample, so without
        # a pause the interrupt would fire every tick and the robot would spin
        # in place instead of searching once and collecting.
        if self.link.now() - self.last_interrupt_at < self.cfg.interrupt_pause_sec:
            return False

        radius = max(0.35, min(0.9, 1.6 * (1.0 - signal)))
        plan = Plan(
            plan_id=self._next_id(),
            subgoals=[Subgoal(type='search_around',
                              x=round(pose[0], 2), y=round(pose[1], 2),
                              radius=round(radius, 2)),
                      Subgoal(type='collect')],
            explanation=(f'сигнал датчика {signal:.2f}: образец рядом, '
                         'ищу на месте'),
        )
        self.link.log.warn(
            f'сигнал {signal:.2f}, текущая подцель {subgoal.describe()} — '
            'прерываю план, образец рядом')
        self.last_interrupt_at = self.link.now()
        self._publish(plan, source='signal')
        return True

    def _current_subgoal(self) -> Subgoal | None:
        """The subgoal the executor is on, taken from the plan we sent."""
        status = self.link.status
        if not status or status.get('plan_id') != self.inflight:
            return None
        index = int(status.get('index') or 0) - 1
        for sent in self.sent_subgoals:
            if sent[0] == self.inflight and sent[1] == index:
                return sent[2]
        return None

    def _budget(self) -> dict[str, float]:
        """What may be spent away from the base, and the floor below it."""
        battery = self.link.battery()
        return_cost = self.link.return_cost()
        spendable = spendable_budget(battery, return_cost)
        return {
            'floor': round(spendable, 1),
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
        )
        try:
            self.client.complete_json(PLANNER_SYSTEM, prompt, tag='probe')
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
        return self.link.now() - self.last_plan_at >= self.cfg.replan_period_sec

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
        self.link.publish_plan(plan.to_wire())
        self.inflight = plan.plan_id
        self.last_plan_at = self.link.now()
        self.force_replan = False
        self.sent_subgoals = [(plan.plan_id, index, subgoal)
                              for index, subgoal in enumerate(plan.subgoals)]
        # Any plan that ends at the base is a decision to stop exploring, so
        # the next one that would do the same thing is the loop to guard
        # against, whether the model wrote it or the budget forced it.
        self._last_was_return = bool(
            plan.subgoals and plan.subgoals[-1].type == 'return_to_base')
        self.link.log.info(
            f'план {plan.plan_id} ({source}) за '
            f'{self.link.now() - self.last_plan_at:.0f} с модели: '
            f'{[item.describe() for item in plan.subgoals]}'
        )
        if plan.explanation:
            self.link.log.info(f'  {plan.explanation}')
        # No journal entry here on purpose: the agent's dashboard already turns
        # a plan's "explanation" into a `decision` entry, so writing one too
        # would show every plan twice in the feed.

    def _next_id(self) -> str:
        self.plan_counter += 1
        return f'llm-{self.plan_counter:03d}'

    def summary(self) -> str:
        return json.dumps(self.stats(), ensure_ascii=False)