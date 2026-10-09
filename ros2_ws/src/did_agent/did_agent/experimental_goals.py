"""Isolated goal-budget experiment; no ROS connections or execution side effects.

The map must contain the agent's observations only.  The budget is a conservative
heuristic, not a calibrated probability of returning.  In particular, search
distance is a configured estimate, not a bound enforced by the existing skill.
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import asdict, dataclass
import hashlib
import json
from math import hypot, isfinite
import re
from typing import Any, Iterable, Sequence

import numpy as np

from did_agent.costmap import CostMap
from did_agent.plan import WORLD_LIMIT, parse_plan
from did_agent.planner import plan_waypoints
from did_judge.judge_model import (
    COLLISION_PENALTY, FALSE_COLLECT_PENALTY, FINISH_REWARD,
    HAZARD_HIT_PENALTY, SAMPLE_REWARD,
)


class GoalValidationError(ValueError):
    """A request needs a new snapshot or a different, feasible candidate."""


def _number(value: Any, name: str, minimum: float | None = None) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not isfinite(value):
        raise GoalValidationError(f'{name} must be a finite number')
    if minimum is not None and value < minimum:
        raise GoalValidationError(f'{name} must be >= {minimum:g}')
    return float(value)


def _integer(value: Any, name: str) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise GoalValidationError(f'{name} must be a non-negative integer')


@dataclass(frozen=True)
class Observation:
    episode_id: str
    revision: int
    pose: tuple[float, float]
    battery: float
    time: float = 0.0
    collected: int = 0
    samples_total: int = 1


@dataclass(frozen=True)
class SearchTarget:
    goal_id: str
    x: float
    y: float
    radius: float = 0.6


@dataclass(frozen=True)
class BudgetPolicy:
    energy_per_meter: float = 1.0
    search_distance: float = 3.0
    return_factor: float = 1.4
    reserve: float = 8.0
    pessimism: float = 0.5


class CandidateBackend:
    """Compute choices and accept an ID against the last private snapshot.

    ``build`` returns a disposable JSON copy.  Modifying it cannot change the
    targets compiled by ``accept``.  A successful acceptance consumes the batch.
    The caller owns the execution lifecycle and must not request a new batch
    while a previous plan is running.
    """

    def __init__(self, costmap: CostMap, base: tuple[float, float],
                 policy: BudgetPolicy | None = None) -> None:
        self.costmap = costmap
        self.base = tuple(base)
        self.policy = policy or BudgetPolicy()
        self._latest: dict[str, Any] | None = None
        self._targets: tuple[SearchTarget, ...] = ()
        self._validate_policy()
        self._validate_point(self.base, 'base')

    def _validate_policy(self) -> None:
        for name, value in asdict(self.policy).items():
            _number(value, name, 0.0)
        if self.policy.energy_per_meter <= 0 or self.policy.search_distance <= 0:
            raise GoalValidationError('energy_per_meter and search_distance must be positive')
        if self.policy.return_factor < 1:
            raise GoalValidationError('return_factor must be >= 1')

    def _validate_point(self, point: Sequence[float], name: str) -> None:
        if not isinstance(point, (tuple, list)) or len(point) != 2:
            raise GoalValidationError(f'{name} must contain x and y')
        for value in point:
            _number(value, name)
            if abs(value) > WORLD_LIMIT:
                raise GoalValidationError(f'{name} is outside the plan coordinate limits')
        if not self.costmap.grid.in_bounds(*self.costmap.world_to_cell(*point)):
            raise GoalValidationError(f'{name} is outside the map')

    def _validate(self, observation: Observation, targets: Sequence[SearchTarget]) -> None:
        self._validate_policy()
        self._validate_point(self.base, 'base')
        if not isinstance(observation.episode_id, str) or not observation.episode_id.strip():
            raise GoalValidationError('episode_id must be a non-empty string')
        for name in ('revision', 'collected', 'samples_total'):
            _integer(getattr(observation, name), name)
        if observation.collected > observation.samples_total:
            raise GoalValidationError('collected cannot exceed samples_total')
        self._validate_point(observation.pose, 'pose')
        _number(observation.battery, 'battery', 0)
        _number(observation.time, 'time', 0)
        ids = {'home'}
        for target in targets:
            if not isinstance(target.goal_id, str) or not re.fullmatch(r'[A-Za-z0-9_-]{1,64}', target.goal_id):
                raise GoalValidationError('goal_id must use 1–64 letters, numbers, underscores or hyphens')
            if target.goal_id in ids:
                raise GoalValidationError(f'duplicate or reserved goal_id: {target.goal_id}')
            ids.add(target.goal_id)
            self._validate_point((target.x, target.y), f'target {target.goal_id}')
            radius = _number(target.radius, 'radius')
            if not 0.1 <= radius <= 3.0:
                raise GoalValidationError('radius must be between 0.1 and 3.0 m')
        for name in ('terrain', 'wall_cost', 'last_seen', 'blocked'):
            layer = getattr(self.costmap, name)
            if layer.shape != self.costmap.grid.shape or not np.isfinite(layer).all():
                raise GoalValidationError(f'invalid map layer: {name}')
        if (self.costmap.terrain <= 0).any() or (self.costmap.wall_cost < 0).any():
            raise GoalValidationError('terrain must be positive and wall_cost non-negative')

    def _snapshot(self, observation: Observation, targets: Sequence[SearchTarget]) -> str:
        metadata = {
            'observation': asdict(observation), 'targets': [asdict(t) for t in targets],
            'policy': asdict(self.policy), 'base': self.base,
            'grid': [self.costmap.resolution, self.costmap.grid.resolution, self.costmap.grid.origin_x,
                     self.costmap.grid.origin_y, self.costmap.grid.shape],
        }
        digest = hashlib.sha256(json.dumps(metadata, sort_keys=True, allow_nan=False).encode())
        # Small terrain drifts intentionally need not bump CostMap.version.
        for name in ('terrain', 'wall_cost', 'last_seen', 'blocked'):
            layer = np.ascontiguousarray(getattr(self.costmap, name))
            digest.update(name.encode())
            digest.update(layer.dtype.str.encode())
            digest.update(layer.tobytes())
        return digest.hexdigest()

    def _energy(self, start: tuple[float, float], goal: tuple[float, float], now: float) -> float | None:
        # Existing plan_waypoints snaps endpoints; a target must itself be free.
        if not all(self.costmap.is_free(*self.costmap.world_to_cell(*p)) for p in (start, goal)):
            return None
        if start == goal:
            return 0.0
        route = plan_waypoints(self.costmap, start, goal, snap_radius=0.0)
        if route is None:
            return None
        energy = self.policy.energy_per_meter * self.costmap.energy_cost(
            [start, *route, goal], pessimism=self.policy.pessimism, now=now,
        )
        if not isfinite(energy):
            raise GoalValidationError('route energy estimate is not finite')
        return energy

    def _search_energy(self, target: SearchTarget, now: float) -> float:
        mask = self.costmap.cells_in_region({'circle': {
            'x': target.x, 'y': target.y, 'r': target.radius,
        }}) & ~self.costmap.blocked
        # A sub-cell disk may miss all cell centres; its containing cell counts.
        mask[self.costmap.world_to_cell(target.x, target.y)] = True
        prices = self.costmap.terrain[mask].astype(float)
        age = np.clip((now - self.costmap.last_seen[mask]) / 90.0, 0.0, 1.5)
        prices = np.where(prices > 1.0,
                          1.0 + (prices - 1.0) * (1.0 + self.policy.pessimism + age), prices)
        energy = float(prices.max()) * self.policy.search_distance * self.policy.energy_per_meter
        if not isfinite(energy):
            raise GoalValidationError('search energy estimate is not finite')
        return energy

    def build(self, observation: Observation, targets: Sequence[SearchTarget]) -> dict[str, Any]:
        """Evaluate observed targets; no sample positions or future events enter."""
        targets = tuple(targets)
        self._validate(observation, targets)
        snapshot_id = self._snapshot(observation, targets)
        candidates = []
        home_energy = self._energy(observation.pose, self.base, observation.time)
        for target in (*targets, SearchTarget('home', *self.base, radius=0.0)):
            home = target.goal_id == 'home'
            energy_to = home_energy if home else self._energy(
                observation.pose, (target.x, target.y), observation.time)
            energy_home = 0.0 if home else self._energy(
                (target.x, target.y), self.base, observation.time)
            reachable = energy_to is not None and energy_home is not None
            energy_search = 0.0 if home else self._search_energy(target, observation.time)
            required = None
            if reachable:
                required = (self.policy.return_factor * energy_to + self.policy.reserve
                            if home else energy_to + energy_search
                            + self.policy.return_factor * energy_home + self.policy.reserve)
                if not isfinite(required):
                    raise GoalValidationError('energy estimate is not finite')
            finished = not home and observation.collected == observation.samples_total
            feasible = reachable and required <= observation.battery and not finished
            reason = ('all samples already collected' if finished else
                      'no free route to goal and base' if not reachable else
                      'insufficient battery including return reserve' if not feasible else
                      'within estimated energy budget')
            candidates.append({
                'goal_id': target.goal_id, 'kind': 'return' if home else 'search',
                'x': target.x, 'y': target.y, 'radius': target.radius,
                'reachable': reachable, 'feasible': feasible, 'reason': reason,
                'energy_to_goal': energy_to, 'energy_search': energy_search,
                'energy_home': energy_home, 'required_battery': required,
                'expected_score': None,
            })
        batch = {
            'snapshot_id': snapshot_id, 'episode_id': observation.episode_id,
            'revision': observation.revision, 'battery': observation.battery,
            'objective': {
                'formula': (f'{SAMPLE_REWARD:g}*N + {FINISH_REWARD:g}*R '
                            f'- {COLLISION_PENALTY:g}*C - {FALSE_COLLECT_PENALTY:g}*F '
                            f'- {HAZARD_HIT_PENALTY:g}*H'),
                'weights': {'N': SAMPLE_REWARD, 'R': FINISH_REWARD,
                            'C': -COLLISION_PENALTY, 'F': -FALSE_COLLECT_PENALTY,
                            'H': -HAZARD_HIT_PENALTY},
                'expected_score': None,
                'note': 'Public judge weights; outcome probabilities are unknown. '
                        'No expected score or return probability is claimed.',
            },
            'budget': {**asdict(self.policy),
                       'note': 'Heuristic estimate, not a 95% return guarantee; '
                               'search_distance is not an enforced travel limit.'},
            'candidates': candidates,
        }
        self._latest = deepcopy(batch)
        self._targets = deepcopy(targets)
        return batch

    def accept(self, snapshot_id: str, goal_id: str, current: Observation) -> dict[str, Any]:
        """Validate freshness and compile an internal candidate into a plan."""
        if self._latest is None or snapshot_id != self._latest['snapshot_id']:
            raise GoalValidationError('unknown or consumed snapshot; build a new batch')
        self._validate(current, self._targets)
        if self._snapshot(current, self._targets) != snapshot_id:
            raise GoalValidationError('stale snapshot: observation, budget or map changed')
        if not isinstance(goal_id, str):
            raise GoalValidationError('goal_id must be a string')
        candidate = next((item for item in self._latest['candidates'] if item['goal_id'] == goal_id), None)
        if candidate is None:
            raise GoalValidationError(f'unknown goal_id: {goal_id}')
        if not candidate['feasible']:
            raise GoalValidationError(f'goal {goal_id} rejected: {candidate["reason"]}')
        steps = ([{'type': 'return_to_base'}] if candidate['kind'] == 'return' else [
            {'type': 'goto', 'x': candidate['x'], 'y': candidate['y']},
            {'type': 'search_around', 'x': candidate['x'], 'y': candidate['y'],
             'radius': candidate['radius']},
            {'type': 'collect'},
        ])
        plan = {'plan_id': f'experiment-{snapshot_id[:16]}-{goal_id}', 'subgoals': steps}
        parse_plan(json.dumps(plan, allow_nan=False))
        self._latest = None
        return plan


def _reachable_mask(costmap: CostMap, start: tuple[float, float]) -> np.ndarray:
    """Flood-fill the A* neighbourhood without crossing blocked diagonals."""
    if not isinstance(start, (tuple, list)) or len(start) != 2:
        raise GoalValidationError('start must contain x and y')
    for coordinate in start:
        _number(coordinate, 'start')
    first = costmap.world_to_cell(*start)
    if not costmap.grid.in_bounds(*first):
        raise GoalValidationError('start is outside the map')
    reached = np.zeros(costmap.grid.shape, dtype=bool)
    if not costmap.is_free(*first):
        return reached
    reached[first] = True
    pending = [first]
    while pending:
        row, col = pending.pop()
        for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1),
                       (-1, -1), (-1, 1), (1, -1), (1, 1)):
            neighbour = (row + dr, col + dc)
            if not costmap.is_free(*neighbour) or reached[neighbour]:
                continue
            if dr and dc and (costmap.blocked[row + dr, col]
                              or costmap.blocked[row, col + dc]):
                continue
            reached[neighbour] = True
            pending.append(neighbour)
    return reached


def search_targets(costmap: CostMap, spacing: float = 1.3, limit: int = 5,
                   exclude: Iterable[str] = (),
                   start: tuple[float, float] | None = None) -> list[SearchTarget]:
    """Choose spaced free cell centres deterministically from the supplied map.

    IDs encode cells and remain stable when other targets are excluded.  With
    ``start`` supplied, disconnected cells are removed before spacing or the
    limit are applied.  Without it, ``build`` alone assesses connectivity.
    """
    _number(spacing, 'spacing', 0)
    if spacing <= 0:
        raise GoalValidationError('spacing must be positive')
    _integer(limit, 'limit')
    eligible = ~costmap.blocked if start is None else _reachable_mask(costmap, start)
    excluded = set(exclude)
    excluded_points = []
    for goal_id in excluded:
        match = re.fullmatch(r'cell_(\d+)_(\d+)', str(goal_id))
        if match:
            cell = tuple(map(int, match.groups()))
            if costmap.grid.in_bounds(*cell):
                excluded_points.append(costmap.cell_to_world(*cell))
    result: list[SearchTarget] = []
    if not limit:
        return result
    for row, col in np.argwhere(eligible):
        goal_id = f'cell_{row}_{col}'
        if goal_id in excluded:
            continue
        x, y = costmap.cell_to_world(int(row), int(col))
        if any(hypot(x - px, y - py) < spacing for px, py in excluded_points):
            continue
        if all(hypot(x - t.x, y - t.y) >= spacing for t in result):
            result.append(SearchTarget(goal_id, x, y))
            if len(result) >= limit:
                break
    return result
