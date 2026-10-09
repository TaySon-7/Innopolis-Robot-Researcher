"""Opt-in goal-selection experiment. No ROS, publishers or automatic API calls.

The model sees a JSON offer and chooses an ID. The backend retains coordinates,
computes energy and validates freshness. Network work is deliberately separate
from the session lifecycle, so a future adapter can run it outside ROS callbacks.
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
import json
from typing import Any, Protocol
from uuid import uuid4

from did_llm.llm_client import LLMUnavailable


SYSTEM = """You select one high-level goal for a sample-search robot.
Use the backend's objective and energy estimates. Only choose a candidate whose
feasible field is true. Prefer useful search while samples remain; return when
search is no longer feasible. Use the offer's observations: the current sensor
value and noise, collection count, and recent attempted goals. A candidate backed
by observed_signal evidence should be preferred to a broad unvisited target when
feasible; do not repeat a failed local search without new observed evidence.
The backend owns these facts; never infer hidden sample positions from goal IDs.
Expected sample yields and return probabilities
are unknown: do not invent them. Energy feasibility is a heuristic, not a proof.
Return exactly {"goal_id": "an offered ID", "reason": "brief explanation"}.
Never output coordinates, velocities, a new plan, or altered costs.
"""


class ChoiceRejected(ValueError):
    """An invalid model decision; suitable for a bounded repair request."""


class SessionError(ValueError):
    """An overlapping, stale or mismatched lifecycle event."""


class JSONClient(Protocol):
    def complete_json(self, system: str, user: str, **kwargs: Any) -> dict[str, Any]: ...


class GoalBackend(Protocol):
    def build(self, observation: Any, targets: Any) -> dict[str, Any]: ...
    def accept(self, snapshot_id: str, goal_id: str, current: Any) -> dict[str, Any]: ...


def _feasible(batch: dict[str, Any]) -> list[dict[str, Any]]:
    return [c for c in batch['candidates'] if c['feasible'] is True]


def parse_choice(data: Any, batch: dict[str, Any]) -> dict[str, str]:
    if not isinstance(data, dict) or set(data) != {'goal_id', 'reason'}:
        raise ChoiceRejected('Return exactly goal_id and reason; no other fields.')
    if not isinstance(data['goal_id'], str):
        raise ChoiceRejected('goal_id must be a string.')
    if data['goal_id'] not in {c['goal_id'] for c in _feasible(batch)}:
        raise ChoiceRejected('goal_id must identify an offered feasible candidate.')
    if not isinstance(data['reason'], str) or not 1 <= len(data['reason'].strip()) <= 500:
        raise ChoiceRejected('reason must contain 1–500 characters.')
    return dict(data)


def choice_response_format(batch: dict[str, Any]) -> dict[str, Any]:
    ids = [c['goal_id'] for c in _feasible(batch)]
    if not ids:
        raise ChoiceRejected('No feasible candidate; do not issue a motion plan.')
    return {'type': 'json_schema', 'json_schema': {
        'name': 'experimental_goal_choice', 'strict': True,
        'schema': {
            'type': 'object', 'additionalProperties': False,
            'required': ['goal_id', 'reason'],
            'properties': {
                'goal_id': {'type': 'string', 'enum': ids},
                'reason': {'type': 'string', 'minLength': 1, 'maxLength': 500},
            },
        },
    }}


def budget_choice(batch: dict[str, Any]) -> dict[str, str]:
    """Explicit baseline: cheapest feasible search, otherwise feasible home.

    This optimizes the energy proxy only, not expected mission score.
    """
    feasible = _feasible(batch)
    choices = [c for c in feasible if c['kind'] == 'search'] or feasible
    if not choices:
        raise ChoiceRejected('No feasible candidate; do not issue a motion plan.')
    candidate = min(choices, key=lambda c: (c['required_battery'], c['goal_id']))
    return {'goal_id': candidate['goal_id'],
            'reason': 'Deterministic baseline: lowest estimated budget among feasible searches, otherwise home.'}


class BudgetStub:
    """Offline stand-in, not a DeepSeek response or quality evaluation."""

    def complete_json(self, system: str, user: str, **kwargs: Any) -> dict[str, str]:
        return budget_choice(json.loads(user)['offer'])


@dataclass(frozen=True)
class Selection:
    snapshot_id: str
    choice: dict[str, str]
    source: str
    errors: tuple[str, ...] = ()


def select_goal(batch: dict[str, Any], client: JSONClient) -> Selection:
    """One call and at most one repair; an outage uses the labelled baseline.

    Supply an explicitly configured LLMClient to try DeepSeek. This function
    never loads credentials or creates a network client itself.
    """
    response_format = choice_response_format(batch)
    errors: list[str] = []
    for attempt in range(2):
        request = {'offer': batch, 'validation_errors': errors}
        try:
            answer = client.complete_json(
                SYSTEM, json.dumps(request, ensure_ascii=False, allow_nan=False),
                tag=f'experimental_goal_{attempt}', max_retries=0,
                response_format=response_format,
            )
            choice = parse_choice(answer, batch)
            return Selection(batch['snapshot_id'], choice,
                             'stub' if isinstance(client, BudgetStub) else 'llm', tuple(errors))
        except ChoiceRejected as error:
            errors.append(str(error))
        except LLMUnavailable:
            # Do not echo provider response bodies or credentials into the demo.
            errors.append('Model unavailable; deterministic fallback used.')
            break
    return Selection(batch['snapshot_id'], budget_choice(batch), 'fallback', tuple(errors))


class DecisionSession:
    """Offline coordinator: offer -> choose -> validate -> finish whole plan.

    Single-threaded lifecycle; callers serialize prepare/commit/status/reset.
    Selection may run elsewhere. reset() invalidates its outstanding answer.
    Completing a goto does not complete the plan. Outcome history records an
    attempted target, never claims that an entire search disk was surveyed.
    """

    def __init__(self, backend: GoalBackend) -> None:
        self.backend = backend
        self._session_id = uuid4().hex[:12]
        self._serial = 0
        self._episode: str | None = None
        self._pending: tuple[int, dict[str, Any]] | None = None
        self._active: dict[str, Any] | None = None
        self._next_step = 0
        self._attempted: set[str] = set()
        self._outcomes: list[dict[str, Any]] = []

    @property
    def attempted_goal_ids(self) -> frozenset[str]:
        return frozenset(self._attempted)

    @property
    def outcomes(self) -> list[dict[str, Any]]:
        return deepcopy(self._outcomes)

    @property
    def busy(self) -> bool:
        return self._active is not None or self._pending is not None

    def reset(self, episode_id: str) -> None:
        """Local bookkeeping only. A future live adapter must stop motion first."""
        self._serial += 1
        self._episode = episode_id
        self._pending = self._active = None
        self._next_step = 0
        self._attempted.clear()
        self._outcomes.clear()

    def prepare(self, observation: Any, targets: Any) -> tuple[int, dict[str, Any]]:
        if self.busy:
            raise SessionError('Wait for the whole active plan or cancel the pending request.')
        if self._episode is None:
            self._episode = observation.episode_id
        if self._episode != observation.episode_id:
            raise SessionError('New episode requires explicit session reset.')
        # No repeated failed search loops in this baseline. Retrying a target
        # after changed evidence will need an explicit policy in a live adapter.
        remaining = [t for t in targets if t.goal_id not in self._attempted]
        batch = self.backend.build(observation, remaining)
        if not _feasible(batch):
            raise ChoiceRejected('No feasible candidate; do not issue a motion plan.')
        self._serial += 1
        self._pending = (self._serial, deepcopy(batch))
        return self._serial, deepcopy(batch)

    def cancel_pending(self) -> None:
        self._serial += 1
        self._pending = None

    def commit(self, ticket: int, selection: Selection, current: Any) -> dict[str, Any]:
        if self._pending is None or ticket != self._pending[0]:
            raise SessionError('Stale or cancelled request; prepare a new offer.')
        batch = self._pending[1]
        if selection.snapshot_id != batch['snapshot_id']:
            raise SessionError('Selection belongs to a different snapshot.')
        choice = parse_choice(selection.choice, batch)
        try:
            plan = self.backend.accept(batch['snapshot_id'], choice['goal_id'], current)
        finally:
            # An outdated battery/map requires a fresh offer, not repeated
            # acceptance of the same answer. Wrong IDs may be repaired earlier.
            self._pending = None
        plan['plan_id'] = f'experimental-{self._session_id}-{ticket}'
        self._active = {'plan': plan, 'goal_id': choice['goal_id'],
                        'source': selection.source, 'reason': choice['reason']}
        self._next_step = 0
        return deepcopy(self._active)

    def on_status(self, status: dict[str, Any]) -> bool:
        """Return True only when the active plan terminates. Ignore stale IDs."""
        if self._active is None or status.get('plan_id') != self._active['plan']['plan_id']:
            return False
        index, state = status.get('index'), status.get('state')
        if isinstance(index, bool) or not isinstance(index, int) or index < 0:
            raise SessionError('Status index must be a non-negative integer.')
        if index < self._next_step:
            return False  # duplicate delivery of an already completed subgoal
        if index != self._next_step:
            raise SessionError('Out-of-order status; missing completion of earlier subgoals.')
        if state == 'running':
            return False
        if state not in ('done', 'failed', 'preempted'):
            raise SessionError('Unknown execution status.')
        if state == 'done':
            self._next_step += 1
            if self._next_step < len(self._active['plan']['subgoals']):
                return False
        goal_id = self._active['goal_id']
        if state != 'preempted' and goal_id != 'home':
            self._attempted.add(goal_id)
        self._outcomes.append({
            'goal_id': goal_id, 'state': state, 'index': index,
            'reason': str(status.get('reason', '')), 'source': self._active['source'],
        })
        self._active = None
        return True
