"""The plan format the agent on /agent/plan accepts, and the budget rules.

Everything here mirrors ``did_agent/plan.py`` and ``did_agent/skills.py``
deliberately. The agent publishes a rejection reason when a plan does not
parse, and the handbook says that reason can be fed back to the model, so
the planner has to know the rules before it publishes rather than learn them
from a rejection. Duplicating the rules also keeps this package free of a
dependency on the executor: a planner that cannot run without the code it
plans for cannot be tested against a stub.
"""

from __future__ import annotations

from math import isfinite
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

#: Subgoal vocabulary accepted by the executor. Anything else is rejected
#: with a message the model can read and act on.
SUBGOAL_FIELDS: dict[str, tuple[str, ...]] = {
    'goto': ('x', 'y'),
    'search_around': ('x', 'y', 'radius'),
    'collect': (),
    'return_to_base': (),
}

#: Arena extent, generous box around it. The executor checks against the real
#: map; this only catches a model that has lost the plot.
WORLD_LIMIT = 10.0

#: Ceiling the executor enforces. Kept here so an oversized plan is cut or
#: refused locally instead of coming back as a rejection.
MAX_SUBGOALS = 50

#: How many subgoals to ask for. Their handbook recommends 3-6: few enough to
#: react when the ground changes, long enough not to spend a model call on
#: every step.
TARGET_SUBGOALS = 4

#: Radius bounds for ``search_around``, from the executor.
RADIUS_MIN = 0.1
RADIUS_MAX = 3.0

#: Battery budget, from the agent's own rule: enough to get home with margin.
#: The return estimate comes from the agent's cost map, so it already knows
#: about expensive ground; the margin covers the ground getting worse.
RETURN_MARGIN = 1.4
RETURN_RESERVE = 8.0

BASE_X = -2.0
BASE_Y = -0.5

#: Free arena, from ARCHITECTURE section 3. Used to keep the model inside the
#: room without making the planner compute geometry.
ARENA_X_MIN, ARENA_X_MAX = -2.8, 2.5
ARENA_Y_MIN, ARENA_Y_MAX = -2.5, 2.5


def budget_floor(return_cost: float | None) -> float | None:
    """Battery below which going home is the only sensible plan.

    Returns None when the agent has not reported a return cost yet, in which
    case no floor can be computed and planning continues on judgement.
    """
    if return_cost is None:
        return None
    return return_cost * RETURN_MARGIN + RETURN_RESERVE


def must_return(battery: float, return_cost: float | None) -> bool:
    """Whether the budget forbids anything but going home."""
    floor = budget_floor(return_cost)
    if floor is None:
        return False
    return battery < floor


class Subgoal(BaseModel):
    """One step, in the executor's format."""

    model_config = ConfigDict(extra='forbid')

    type: Literal['goto', 'search_around', 'collect', 'return_to_base']
    x: float = 0.0
    y: float = 0.0
    radius: float = 0.0

    @field_validator('x', 'y')
    @classmethod
    def _finite_and_in_world(cls, value: float) -> float:
        if not isfinite(value) or abs(value) > WORLD_LIMIT:
            raise ValueError(f'coordinate {value} is out of the ±{WORLD_LIMIT} m world')
        return float(value)

    @model_validator(mode='after')
    def _radius_if_needed(self) -> 'Subgoal':
        if self.type == 'search_around' and not RADIUS_MIN <= self.radius <= RADIUS_MAX:
            raise ValueError(
                f'radius must be between {RADIUS_MIN} and {RADIUS_MAX} m, '
                f'got {self.radius}'
            )
        if self.type == 'return_to_base':
            return self.model_copy(update={'x': BASE_X, 'y': BASE_Y})
        return self

    def to_wire(self) -> dict[str, Any]:
        """The JSON object as the executor wants it, with only relevant keys."""
        if self.type == 'goto':
            return {'type': 'goto', 'x': round(self.x, 3), 'y': round(self.y, 3)}
        if self.type == 'search_around':
            return {'type': 'search_around', 'x': round(self.x, 3),
                    'y': round(self.y, 3), 'radius': round(self.radius, 3)}
        return {'type': self.type}

    def describe(self) -> str:
        if self.type == 'search_around':
            return f'search_around({self.x:g}, {self.y:g}, r={self.radius:g})'
        if self.type == 'goto':
            return f'goto({self.x:g}, {self.y:g})'
        return self.type


class Plan(BaseModel):
    """A plan ready to publish."""

    model_config = ConfigDict(extra='forbid')

    plan_id: str = Field(min_length=1)
    subgoals: list[Subgoal] = Field(min_length=1, max_length=MAX_SUBGOALS)
    #: Shown on the dashboard as the reasoning behind the plan.
    explanation: str = ''

    def to_wire(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            'plan_id': self.plan_id,
            'subgoals': [item.to_wire() for item in self.subgoals],
        }
        if self.explanation:
            payload['explanation'] = self.explanation
        return payload


class PlanRejected(ValueError):
    """The model's answer is not a plan. The message goes back to it."""


def parse_model_plan(raw: str | dict[str, Any],
                     plan_id: str) -> Plan:
    """Turn a model answer into a validated plan.

    Errors are phrased as instructions, because the handbook's loop feeds the
    rejection reason straight back into the next prompt: "radius must be
    between 0.1 and 3.0 m" tells the model what to change, "invalid JSON" does
    not.
    """
    if isinstance(raw, str):
        import json
        try:
            data = json.loads(raw)
        except (TypeError, ValueError) as error:
            raise PlanRejected(
                f'ответ не разобран как JSON ({error}). Верни только JSON.'
            ) from error
    else:
        data = raw

    if isinstance(data, list):
        data = {'subgoals': data}
    if not isinstance(data, dict):
        raise PlanRejected('ответ должен быть объектом с полем "subgoals"')

    items = data.get('subgoals')
    if not isinstance(items, list):
        raise PlanRejected('в ответе нет списка "subgoals"')
    if not items:
        raise PlanRejected('список "subgoals" пуст — план без подцелей ничего не делает')
    if len(items) > MAX_SUBGOALS:
        raise PlanRejected(f'подцелей {len(items)}, максимум {MAX_SUBGOALS}')

    subgoals: list[Subgoal] = []
    for index, item in enumerate(items):
        if not isinstance(item, dict):
            raise PlanRejected(f'подцель {index}: ожидался объект, получено {item!r}')
        kind = item.get('type')
        if kind not in SUBGOAL_FIELDS:
            known = ', '.join(SUBGOAL_FIELDS)
            raise PlanRejected(
                f'подцель {index}: неизвестный тип {kind!r}. Допустимы: {known}'
            )
        # Fields the type does not use are dropped rather than rejected, so a
        # model that fills every field in does not lose the whole plan.
        payload = {key: item[key] for key in SUBGOAL_FIELDS[kind]
                   if key in item}
        if kind == 'search_around' and 'radius' not in payload:
            raise PlanRejected(
                f'подцель {index}: search_around требует radius '
                f'от {RADIUS_MIN} до {RADIUS_MAX} м'
            )
        try:
            subgoals.append(Subgoal.model_validate({'type': kind, **payload}))
        except Exception as error:  # noqa: BLE001 - message is for the model
            raise PlanRejected(f'подцель {index}: {_brief(error)}') from error

    explanation = data.get('explanation', '')
    if not isinstance(explanation, str):
        explanation = ''

    return Plan(plan_id=plan_id, subgoals=subgoals, explanation=explanation)


def home_plan(plan_id: str, explanation: str = '') -> Plan:
    """The fallback: go home and finish.

    Used when the budget forbids anything else, when the model is unavailable,
    or when its answer did not parse. It has to be a plan rather than nothing
    so the episode still ends with the robot on the base.
    """
    return Plan(
        plan_id=plan_id,
        subgoals=[Subgoal(type='return_to_base')],
        explanation=explanation or 'батареи не хватает на обход, возвращаюсь',
    )


def _brief(error: Exception) -> str:
    """First line of a pydantic error, without the noisy model dump."""
    text = str(error).strip().splitlines()
    return text[0] if text else error.__class__.__name__