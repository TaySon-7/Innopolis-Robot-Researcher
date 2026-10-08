"""Parsing and validation of the plan that the LLM planner sends (/agent/plan)."""

from __future__ import annotations

from dataclasses import dataclass
from dataclasses import field
from math import isfinite
import json
from typing import Any

SUBGOAL_FIELDS = {
    'goto': ('x', 'y'),
    'search_around': ('x', 'y', 'radius'),
    'collect': (),
    'return_to_base': (),
}

# A generous box around the arena; the real check is done against the map.
WORLD_LIMIT = 10.0
MAX_SUBGOALS = 50


class PlanError(ValueError):
    """The plan cannot be executed; the message goes back to the planner."""


@dataclass
class Subgoal:
    """One step of a plan."""

    type: str
    x: float = 0.0
    y: float = 0.0
    radius: float = 0.0

    def describe(self) -> str:
        if self.type in ('goto', 'search_around'):
            extra = f', r={self.radius:g}' if self.type == 'search_around' else ''
            return f'{self.type}({self.x:g}, {self.y:g}{extra})'
        return self.type


@dataclass
class Plan:
    """A parsed plan."""

    plan_id: str
    subgoals: list[Subgoal] = field(default_factory=list)


def _number(item: dict[str, Any], key: str, index: int) -> float:
    value = item.get(key)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise PlanError(f'subgoal {index}: field {key!r} must be a number, got {value!r}')
    if not isfinite(value) or abs(value) > WORLD_LIMIT:
        raise PlanError(f'subgoal {index}: field {key!r}={value!r} is out of range')
    return float(value)


def parse_plan(text: str) -> Plan:
    """Parse the JSON text of a plan; raise PlanError with a clear reason."""
    try:
        data = json.loads(text)
    except (TypeError, ValueError) as error:
        raise PlanError(f'plan is not valid JSON: {error}') from error
    if isinstance(data, list):  # tolerate a bare list of subgoals
        data = {'subgoals': data}
    if not isinstance(data, dict):
        raise PlanError('plan must be a JSON object with a "subgoals" list')
    items = data.get('subgoals')
    if not isinstance(items, list):
        raise PlanError('plan has no "subgoals" list')
    if len(items) > MAX_SUBGOALS:
        raise PlanError(f'plan has {len(items)} subgoals, the limit is {MAX_SUBGOALS}')

    subgoals = []
    for index, item in enumerate(items):
        if not isinstance(item, dict):
            raise PlanError(f'subgoal {index}: must be an object, got {item!r}')
        kind = item.get('type')
        if kind not in SUBGOAL_FIELDS:
            known = ', '.join(SUBGOAL_FIELDS)
            raise PlanError(f'subgoal {index}: unknown type {kind!r} (known: {known})')
        values = {key: _number(item, key, index) for key in SUBGOAL_FIELDS[kind]}
        if kind == 'search_around' and not 0.1 <= values['radius'] <= 3.0:
            raise PlanError(f'subgoal {index}: radius must be between 0.1 and 3.0 m')
        subgoals.append(Subgoal(type=kind, **values))
    return Plan(plan_id=str(data.get('plan_id', 'plan')), subgoals=subgoals)
