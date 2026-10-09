"""Live goal selection over the agent's public, precomputed goal offers.

This module has no ROS or executor imports. The agent owns routes, energy and
canonical subgoals; the model can only select an offered ID. One request runs in
a worker and one whole plan runs at a time. Operator commands, episode changes
and replacement offers invalidate late model answers without interrupting a
search just because the next model response arrived.
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, field
from math import isfinite
from threading import Event, Thread
from typing import Any
from uuid import uuid4

from did_llm.experimental_selector import (
    BudgetStub, ChoiceRejected, JSONClient, Selection, budget_choice, parse_choice, select_goal,
)
from did_llm.llm_client import LLMUnavailable


@dataclass
class _Request:
    generation: int
    episode_id: int
    offer: dict[str, Any]
    done: Event = field(default_factory=Event)
    result: Selection | None = None


@dataclass
class _Active:
    plan_id: str
    snapshot_id: str
    published_at: float
    acknowledged: bool = False


class MathPlanner:
    """Sequential, asynchronous planner using only the published JSON contract.

    ``link`` supplies state, status, command/command_pending, now(), state_age(),
    publish_plan() and journal(). Each snapshot is attempted once. Rejected or
    unacknowledged submissions wait for a new backend offer instead of repeatedly
    publishing a plan that could preempt the executor.

    ``selection_policy='budget'`` is the matched deterministic baseline: it
    uses the same offers, validation and execution lifecycle without a model.
    """

    def __init__(self, link: Any, client: JSONClient | None = None, *,
                 selection_policy: str = 'llm', state_max_age: float = 5.0,
                 ack_timeout: float = 10.0) -> None:
        if selection_policy not in ('llm', 'budget'):
            raise ValueError('selection_policy must be llm or budget')
        if selection_policy == 'llm' and client is None:
            raise ValueError('selection_policy=llm requires a model client')
        self.link = link
        self.client = client
        self.selection_policy = selection_policy
        self.state_max_age = state_max_age
        self.ack_timeout = ack_timeout
        self._generation = 0
        self._episode_id: int | None = None
        self._request: _Request | None = None
        self._active: _Active | None = None
        self._consumed: set[str] = set()
        self._feedback: str | None = None
        self._held = False
        self._hold_observed = False
        self._serial = 0
        self._session = uuid4().hex[:12]

    @property
    def inflight(self) -> str | None:
        return self._active.plan_id if self._active else None

    @property
    def busy(self) -> bool:
        return self._active is not None or self._request is not None

    def metrics(self) -> dict[str, Any]:
        """Sanitized run counters, without requests, responses or credentials."""
        counters = dict.fromkeys(('calls_made', 'cache_hits', 'blocked_by_budget', 'failed'), 0)
        model = None
        if self.selection_policy == 'llm' and self.client is not None:
            read_stats = getattr(self.client, 'stats', None)
            stats = read_stats() if callable(read_stats) else {}
            for key in counters:
                value = stats.get(key)
                if isinstance(value, int) and not isinstance(value, bool) and value >= 0:
                    counters[key] = value
            name = getattr(getattr(self.client, 'cfg', None), 'model', None)
            if isinstance(name, str):
                model = name
        return {'selection_policy': self.selection_policy, 'model': model,
                'client_stats': counters}

    def _journal(self, title: str, text: str = '', *, status: str = 'open',
                 **extra: Any) -> None:
        self.link.journal('llm', title, text, status=status,
                          source='llm_math_planner', **extra)

    def _invalidate(self) -> None:
        # The worker cannot be cancelled mid-HTTP call. Keep its slot until it
        # finishes, but fence its answer and subsequent repairs by generation.
        self._generation += 1

    def _commands(self) -> None:
        if not getattr(self.link, 'command_pending', False):
            return
        self.link.command_pending = False
        command = getattr(self.link, 'command', None)
        if command not in ('stop', 'auto', 'autonomous', 'manual', 'llm'):
            return
        self._invalidate()
        self._active = None
        self._held = command != 'llm'
        self._hold_observed = False

    def _state(self) -> dict[str, Any] | None:
        state = self.link.state
        age = self.link.state_age()
        if (not isinstance(state, dict) or not isinstance(age, (int, float))
                or not isfinite(age) or not 0 <= age <= self.state_max_age):
            return None
        episode = state.get('episode_id')
        if isinstance(episode, bool) or not isinstance(episode, int):
            return None
        if episode != self._episode_id:
            self._invalidate()
            self._episode_id = episode
            self._active = None
            self._consumed.clear()
            self._feedback = None
            self._held = False
            self._hold_observed = False
        return state

    @staticmethod
    def _offer(state: dict[str, Any]) -> dict[str, Any] | None:
        offer = state.get('goal_offer')
        if (not isinstance(offer, dict)
                or not isinstance(offer.get('snapshot_id'), str)
                or not offer['snapshot_id']
                or str(offer.get('episode_id')) != str(state['episode_id'])
                or not isinstance(offer.get('candidates'), list)):
            return None
        feasible = [c for c in offer['candidates']
                    if isinstance(c, dict) and c.get('feasible') is True]
        if (any(not isinstance(c, dict) or not isinstance(c.get('goal_id'), str)
                or not isinstance(c.get('feasible'), bool) for c in offer['candidates'])
                or len({c['goal_id'] for c in offer['candidates']}) != len(offer['candidates'])):
            return None
        battery = offer.get('battery')
        if (isinstance(battery, bool) or not isinstance(battery, (int, float))
                or not isfinite(battery) or battery < 0):
            return None
        for candidate in feasible:
            steps = candidate.get('subgoals')
            if (not isinstance(candidate.get('goal_id'), str)
                    or not isinstance(steps, list) or not steps
                    or any(not isinstance(step, dict)
                           or step.get('type') not in
                           ('goto', 'search_around', 'collect', 'return_to_base')
                           for step in steps)):
                return None
            for key in ('required_battery', 'energy_to_goal', 'energy_search', 'energy_home'):
                value = candidate.get(key)
                if (isinstance(value, bool) or not isinstance(value, (int, float))
                        or not isfinite(value) or value < 0):
                    return None
        return offer

    def _terminal(self, state: dict[str, Any], offer: dict[str, Any] | None) -> None:
        active = self._active
        if active is None:
            return
        status = self.link.status
        if isinstance(status, dict) and status.get('plan_id') == active.plan_id:
            active.acknowledged = True
            rejected = (isinstance(status.get('data'), dict)
                        and status['data'].get('accepted') is False)
            if rejected:
                self._feedback = str(status.get('reason') or 'Backend rejected the choice.')[:500]
                self._journal('Бэкенд отклонил выбор цели', self._feedback,
                              status='rejected', plan_id=active.plan_id)
                self._active = None
                return
            if (status.get('plan_complete') is True
                    and status.get('state') in ('done', 'failed', 'preempted')):
                self._journal('План завершён', str(status.get('reason') or ''),
                              status='confirmed' if status['state'] == 'done' else 'rejected',
                              plan_id=active.plan_id, execution_state=status['state'])
                self._active = None
                return
        current = state.get('current') or {}
        if (offer is not None and offer['snapshot_id'] != active.snapshot_id
                and current.get('state') != 'running'):
            # Even if the terminal status was dropped, a replacement offer
            # explicitly certifies idle execution in the backend contract.
            self._active = None
            return
        if current.get('plan_id') == active.plan_id:
            active.acknowledged = True
        elif current.get('state') == 'running':
            # Another accepted plan owns the executor. Never preempt it.
            self._active = None
            self._invalidate()
            return
        if not active.acknowledged and self.link.now() - active.published_at >= self.ack_timeout:
            self._feedback = 'Previous plan was not acknowledged; use a fresh backend offer.'
            self._journal('Нет подтверждения приёма плана', self._feedback,
                          status='rejected', plan_id=active.plan_id)
            self._active = None

    @staticmethod
    def _forced_return(state: dict[str, Any], offer: dict[str, Any]) -> Selection | None:
        feasible = [c for c in offer['candidates'] if c.get('feasible') is True]
        returns = [c for c in feasible if c.get('kind') == 'return']
        observations = offer.get('observations') or state
        collected = observations.get('collected')
        total = observations.get('samples_total')
        complete = (isinstance(collected, int) and not isinstance(collected, bool)
                    and isinstance(total, int) and not isinstance(total, bool)
                    and 0 <= total <= collected)
        emergency = [c for c in returns if c.get('emergency') is True]
        if returns and (emergency or complete or not any(c.get('kind') != 'return' for c in feasible)):
            chosen = min(emergency or returns,
                         key=lambda c: (c['required_battery'], c['goal_id']))
            reason = ('Emergency return: search budget is exhausted.' if emergency else
                      'All samples collected; return to base.' if complete else
                      'Only return is feasible within the backend energy budget.')
            return Selection(offer['snapshot_id'],
                             {'goal_id': chosen['goal_id'], 'reason': reason}, 'budget')
        return None

    def _publish(self, selection: Selection, offer: dict[str, Any]) -> None:
        try:
            choice = parse_choice(selection.choice, offer)
        except ChoiceRejected:
            self._journal('Ответ не соответствует предложенным целям', status='rejected')
            return
        if selection.snapshot_id != offer['snapshot_id']:
            return
        candidate = next(c for c in offer['candidates'] if c['goal_id'] == choice['goal_id'])
        source = selection.source if selection.source in ('llm', 'budget') else 'fallback'
        self._serial += 1
        plan_id = f'math-{self._session}-{self._serial}'
        decision = {key: candidate[key] for key in (
            'goal_id', 'required_battery', 'energy_to_goal', 'energy_search', 'energy_home')}
        decision['battery'] = offer['battery']
        payload = {
            'plan_id': plan_id, 'source': source, 'explanation': choice['reason'],
            'subgoals': deepcopy(candidate['subgoals']),
            'goal_selection': {'snapshot_id': offer['snapshot_id'], 'goal_id': choice['goal_id']},
            'decision': decision,
        }
        # Latch before publishing: the next state can lag behind the command.
        self._active = _Active(plan_id, offer['snapshot_id'], self.link.now())
        self.link.publish_plan(payload)
        self._feedback = None
        self._journal('Выбрана цель по расчёту бэкенда', choice['reason'],
                      status='confirmed', plan_id=plan_id, decision_source=source,
                      decision=deepcopy(decision), errors=list(selection.errors))

    def _start(self, offer: dict[str, Any]) -> None:
        batch = deepcopy(offer)
        if self._feedback:
            batch['feedback'] = {'previous_rejection': self._feedback}
        request = _Request(self._generation, self._episode_id, batch)
        self._request = request

        planner = self

        class CurrentClient:
            def complete_json(self, system: str, user: str, **kwargs: Any) -> dict[str, Any]:
                if request.generation != planner._generation:
                    raise LLMUnavailable('Cancelled planner generation.', reason='cancelled')
                return planner.client.complete_json(system, user, **kwargs)

        def choose() -> None:
            try:
                selected = select_goal(batch, CurrentClient())
                request.result = (Selection(selected.snapshot_id, selected.choice, 'fallback',
                                            selected.errors)
                                  if isinstance(self.client, BudgetStub) else selected)
            except Exception:  # Untrusted provider failure must not kill the loop or leak its body.
                try:
                    request.result = Selection(
                        batch['snapshot_id'], budget_choice(batch), 'fallback',
                        ('Model selection failed; deterministic fallback used.',))
                except (ChoiceRejected, KeyError, TypeError, ValueError):
                    request.result = None
            finally:
                request.done.set()

        Thread(target=choose, name='math-goal-selection', daemon=True).start()
        self._journal('Модель выбирает цель', snapshot_id=offer['snapshot_id'])

    def tick(self) -> None:
        """Cheap timer callback. All network calls and repairs run off-thread."""
        state = self._state()
        # New-episode initialization clears a previous run's operator hold, but
        # an explicit stop received in this same tick still wins.
        self._commands()
        if state is None:
            return
        mode = state.get('control_mode')
        if mode not in ('llm', 'fallback'):
            if self._held:
                self._hold_observed = True
            if self._request and self._request.generation == self._generation:
                self._invalidate()
            self._active = None
            return
        if self._held and self._hold_observed:
            # Backend switched from an observed held mode to LLM, e.g. a new
            # explicit start whose command was sent before this node subscribed.
            self._held = False
        if self._held or state.get('finished') is True:
            if self._request and self._request.generation == self._generation:
                self._invalidate()
            self._active = None
            return
        offer = self._offer(state)
        self._terminal(state, offer)
        if self._request and (offer is None or
                             self._request.offer['snapshot_id'] != offer['snapshot_id']):
            if self._request.generation == self._generation:
                self._invalidate()
        completed = None
        if self._request and self._request.done.is_set():
            completed, self._request = self._request, None
        if self._active is not None or (state.get('current') or {}).get('state') == 'running':
            return
        if offer is None:
            return
        if (completed is not None and completed.generation == self._generation
                and completed.episode_id == self._episode_id
                and completed.offer['snapshot_id'] == offer['snapshot_id']):
            if completed.result is not None:
                self._publish(completed.result, offer)
            return
        token = offer['snapshot_id']
        if token in self._consumed:
            return
        forced = self._forced_return(state, offer)
        if forced is not None:
            self._consumed.add(token)
            self._publish(forced, offer)
            return
        if self._request is not None:
            return  # At most one HTTP request, including an invalidated worker.
        self._consumed.add(token)
        if not any(c.get('feasible') is True for c in offer['candidates']):
            self._journal('Нет допустимой цели',
                          'Бэкенд не нашёл маршрут в доступном энергобюджете.', status='rejected')
            return
        if self.selection_policy == 'budget':
            self._publish(Selection(token, budget_choice(offer), 'budget'), offer)
            return
        self._start(offer)
