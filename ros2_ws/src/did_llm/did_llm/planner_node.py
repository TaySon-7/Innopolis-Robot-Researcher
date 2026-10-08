"""The planning loop: state in, plan out.

The order of the checks matters more than the model call. Before asking
anything, the loop settles what can be decided without a model: the episode is
over, the budget forbids going out, or the API is down. Each of those has a
deterministic answer, and spending a model call to rediscover it would make
the planner both slower and less reliable.
"""

from __future__ import annotations

import json
from typing import Any

from did_llm.agent_link import AgentLink
from did_llm.agent_plan import (
    Plan,
    PlanRejected,
    home_plan,
    must_return,
    parse_model_plan,
)
from did_llm.llm_client import LLMClient, LLMUnavailable
from did_llm.prompts import (
    PLANNER_SYSTEM,
    build_planner_prompt,
)


class PlannerConfig:
    """Knobs the node exposes as ROS parameters."""

    def __init__(self, mission: str = 'Собрать образцы и вернуться на базу',
                 min_subgoals: int = 3,
                 max_subgoals: int = 6,
                 replan_period_sec: float = 12.0,
                 max_state_staleness_sec: float = 5.0,
                 repair_attempts: int = 1) -> None:
        self.mission = mission
        self.min_subgoals = min_subgoals
        self.max_subgoals = max_subgoals
        self.replan_period_sec = replan_period_sec
        self.max_state_staleness_sec = max_state_staleness_sec
        self.repair_attempts = repair_attempts


class Planner:
    """Decides when to plan and what to publish."""

    def __init__(self, link: AgentLink, client: LLMClient,
                 config: PlannerConfig | None = None) -> None:
        self.link = link
        self.client = client
        self.cfg = config or PlannerConfig()

        self.plan_counter = 0
        self.inflight: str | None = None
        self.inflight_since: float | None = None
        self.last_plan_at: float | None = None
        self.feedback: str = ''
        self.auto_requested = False
        #: Set once the episode has been handed to the agent; after that the
        #: planner must stay out of the way.
        self.handed_over = False
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
            # The agent is driving itself now. Publishing plans on top of its
            # own behaviour would preempt whatever it is doing, every tick,
            # forever.
            return
        if self.link.state is None:
            return
        if self.link.state_age() > self.cfg.max_state_staleness_sec:
            # The agent stopped publishing, which means it is not running or
            # the episode is over. Planning on a stale pose would be a guess.
            return
        if self.link.finished():
            return

        outcome = self._last_outcome()
        if outcome is not None:
            self._note_outcome(outcome)
            return

        if not self.client.cfg.configured:
            self._go_autonomous('API не настроен')
            return

        if must_return(self.link.battery(), self.link.return_cost()):
            if self.last_was_return:
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

        if not self._due():
            return
        self._plan_with_model()

    # ------------------------------------------------------------------ model
    def _plan_with_model(self) -> None:
        prompt = build_planner_prompt(
            self.cfg.mission, self.link.state or {},
            self.link.status, self.feedback,
        )
        try:
            answer = self.client.complete_json(PLANNER_SYSTEM, prompt,
                                               tag='plan')
        except LLMUnavailable as error:
            # The client already retried and gave up. Handing the episode to
            # the agent's own behaviour beats holding a dead plan.
            self._go_autonomous(f'модель недоступна: {error}')
            return

        for attempt in range(self.cfg.repair_attempts + 1):
            try:
                plan = parse_model_plan(answer, self._next_id())
            except PlanRejected as reason:
                self.rejections.append(str(reason))
                self.link.journal(
                    'result',
                    f'План отклонён проверкой: {reason}',
                    status='rejected',
                    source='llm_planner',
                )
                if attempt >= self.cfg.repair_attempts:
                    # Out of repairs: the agent's own policy is a better bet
                    # than a plan we cannot make sense of.
                    self._go_autonomous('модель не вернула корректный план')
                    return
                try:
                    answer = self.client.complete_json(
                        PLANNER_SYSTEM,
                        f'{prompt}\n\nТвой прошлый ответ отклонён: {reason}\n'
                        'Исправь и верни полный JSON заново.',
                        tag='repair',
                    )
                except LLMUnavailable as error:
                    self._go_autonomous(f'модель не ответила: {error}')
                    return
                continue

            self._trim(plan)
            self._publish(plan, source='llm')
            return

    def _trim(self, plan: Plan) -> None:
        """Cut a plan down to the size the loop can react to.

        Trimming keeps the tail, not the head: the head is where the plan
        commits to a direction, and a cut that removes it leaves a plan that
        starts halfway through.
        """
        limit = self.cfg.max_subgoals
        if len(plan.subgoals) <= limit:
            return
        trimmed = Plan(
            plan_id=plan.plan_id,
            subgoals=plan.subgoals[-limit:],
            explanation=plan.explanation,
        )
        plan.subgoals = trimmed.subgoals

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

    # ------------------------------------------------------------------ edges
    def _due(self) -> bool:
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
        self.auto_requested = True
        self.handed_over = True
        self.inflight = None
        self.link.publish_command('auto')
        self.link.journal('decision', f'Переход в автономный режим: {why}')
        self.link.log.warn(f'передача в автономный режим: {why}')

    def _publish(self, plan: Plan, *, source: str) -> None:
        self.link.publish_plan(plan.to_wire())
        self.inflight = plan.plan_id
        self.last_plan_at = self.link.now()
        # Any plan that ends at the base is a decision to stop exploring, so
        # the next one that would do the same thing is the loop to guard
        # against, whether the model wrote it or the budget forced it.
        self._last_was_return = bool(
            plan.subgoals and plan.subgoals[-1].type == 'return_to_base')
        self.link.log.info(
            f'план {plan.plan_id} ({source}): '
            f'{[item.describe() for item in plan.subgoals]}'
        )
        if plan.explanation:
            self.link.log.info(f'  {plan.explanation}')
        self.link.journal(
            'decision',
            f'План {plan.plan_id} ({source})',
            text=plan.explanation or 'причина не указана',
            source='llm_planner',
            subgoals=[item.describe() for item in plan.subgoals],
        )

    def _next_id(self) -> str:
        self.plan_counter += 1
        return f'llm-{self.plan_counter:03d}'

    def summary(self) -> str:
        return json.dumps(self.stats(), ensure_ascii=False)