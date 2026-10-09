"""Production goal offers built only from the robot's observed state.

An offer stays stable while a slow model thinks.  Acceptance recomputes route
and energy on the current map; a changed clock alone does not invalidate it.
This module owns no ROS objects and never reads scenario truth.
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import replace
from math import hypot, isfinite
from typing import Any
from uuid import uuid4

from did_agent.costmap import CostMap
from did_agent.experimental_goals import (
    BudgetPolicy, CandidateBackend, GoalValidationError, Observation, SearchTarget,
    search_targets,
)
from did_agent.planner import plan_waypoints
from did_agent.robot import sample_signal_near


def candidate_steps(candidate: dict[str, Any]) -> list[dict[str, Any]]:
    """Compile coordinates belonging to the backend, never model coordinates."""
    if candidate['kind'] == 'return':
        return [{'type': 'return_to_base'}]
    return [
        {'type': 'goto', 'x': candidate['x'], 'y': candidate['y']},
        {'type': 'search_around', 'x': candidate['x'], 'y': candidate['y'],
         'radius': candidate['radius']},
        {'type': 'collect'},
    ]


class _RuntimeBackend(CandidateBackend):
    def _energy(self, start, goal, now):
        # Navigation can leave an inflated/just-observed blocked start cell.
        # Match its 0.5 m recovery, but never snap a blocked destination.
        if not self.costmap.is_free(*self.costmap.world_to_cell(*goal)):
            return None
        route = plan_waypoints(self.costmap, start, goal, snap_radius=0.5)
        if route is None:
            return None
        if start == goal:
            return 0.0
        energy = self.policy.energy_per_meter * self.costmap.energy_cost(
            [start, *route, goal], pessimism=self.policy.pessimism, now=now)
        if not isfinite(energy):
            raise GoalValidationError('route energy estimate is not finite')
        return energy


class GoalOffers:
    """Single-use choices and whole-plan lifecycle for the live agent."""

    def __init__(self, costmap: CostMap, base: tuple[float, float],
                 policy: BudgetPolicy | None = None) -> None:
        self.costmap = costmap
        self.base = base
        self.policy = policy or BudgetPolicy()
        self._offer: dict[str, Any] | None = None
        self._origin: Observation | None = None
        self._targets: tuple[SearchTarget, ...] = ()
        self._active: dict[str, Any] | None = None
        self._attempted: set[str] = set()
        self._history: list[dict[str, Any]] = []
        self._revision = 0

    @property
    def active(self) -> bool:
        return self._active is not None

    @property
    def offer(self) -> dict[str, Any] | None:
        return deepcopy(self._offer)

    def invalidate(self, *, interrupt: bool = False) -> None:
        """Manual control or reset revokes all outstanding model decisions."""
        self._offer = None
        self._origin = None
        self._targets = ()
        self._revision += 1
        if interrupt:
            self._active = None

    def _stale(self, observation: Observation) -> bool:
        origin = self._origin
        return origin is None or (
            observation.episode_id != origin.episode_id
            or observation.revision != origin.revision
            or observation.collected != origin.collected
            or observation.samples_total != origin.samples_total
            or hypot(observation.pose[0] - origin.pose[0],
                     observation.pose[1] - origin.pose[1]) > 0.2)

    def _evaluate(self, observation: Observation, targets) -> dict[str, Any]:
        batch = _RuntimeBackend(self.costmap, self.base, self.policy).build(observation, targets)
        for candidate in batch['candidates']:
            candidate['emergency'] = False
            if candidate['kind'] == 'return' and candidate['reachable'] and not candidate['feasible']:
                # Once the reserve is breached, returning is still the useful
                # best effort action. This does not promise enough energy.
                candidate['feasible'] = True
                candidate['emergency'] = True
                candidate['reason'] = 'return reserve breached; best effort return, no guarantee'
            candidate['subgoals'] = candidate_steps(candidate)
            candidate['evidence'] = ('observed_signal' if candidate['goal_id'].startswith('local_')
                                     else 'exploration' if candidate['kind'] == 'search' else 'return')
        if any(c['kind'] == 'search' and c['feasible'] for c in batch['candidates']):
            home = next(c for c in batch['candidates'] if c['kind'] == 'return')
            if not home['emergency']:
                home['feasible'] = False
                home['reason'] = 'search remains feasible; continue collecting before finishing'
        return batch

    def build(self, observation: Observation, *, sensor: float = 0.0,
              noise: float = 0.0) -> dict[str, Any] | None:
        """Compute at idle only; repeated calls reuse the exact outstanding offer."""
        if self.active:
            return None
        if self._offer is not None:
            if not self._stale(observation):
                return self.offer
            self.invalidate()
        start_cell = self.costmap.nearest_free(*self.costmap.world_to_cell(*observation.pose), 0.5)
        targets: list[SearchTarget] = []
        if start_cell is not None and observation.collected < observation.samples_total:
            start = self.costmap.cell_to_world(*start_cell)
            # A failed measurement near this same spot is not fresh evidence.
            repeated_local = any(
                item['goal_id'].startswith('local_')
                and item['collected'] == observation.collected
                and hypot(start[0] - item['x'], start[1] - item['y']) < 0.5
                for item in self._history)
            if sample_signal_near(sensor, noise) and not repeated_local:
                local_id = f'local_{start_cell[0]}_{start_cell[1]}_n{observation.collected}'
                radius = max(0.35, min(1.2, 1.5 * (1.0 - sensor) + 0.35))
                targets.append(SearchTarget(local_id, *start, radius))
            # Generate a stable grid first, then remove only attempted IDs.
            # A planned circle never becomes a fictitiously observed region.
            # Match the empirically safer autonomous coverage lattice.  At
            # 1.3 m, pillar inflation can move/remove nominal points and leave
            # pockets that are only heard late in the battery budget.
            pool = search_targets(self.costmap, spacing=1.0, start=start, limit=1000)
            pool = [target for target in pool if target.goal_id not in self._attempted
                    and all(hypot(target.x - item['x'], target.y - item['y']) >= 0.45
                            for item in self._history if not item['goal_id'].startswith('local_'))
                    and all(hypot(target.x - item.x, target.y - item.y) >= 0.5 for item in targets)]
            pool.sort(key=lambda target: hypot(target.x - observation.pose[0], target.y - observation.pose[1]))
            targets.extend(pool[:5 - len(targets)])
        self._targets = tuple(targets)
        batch = self._evaluate(observation, targets)
        batch['snapshot_id'] = uuid4().hex
        batch['revision'] = self._revision
        batch['observations'] = {
            'pose': {'x': observation.pose[0], 'y': observation.pose[1]},
            'sensor': {'value': sensor, 'noise_estimate': noise},
            'collected': observation.collected, 'samples_total': observation.samples_total,
            'attempted_goal_ids': sorted(self._attempted),
            'recent_attempts': deepcopy(self._history[-12:]),
        }
        self._origin = observation
        self._offer = deepcopy(batch)
        return self.offer

    def accept(self, envelope: dict[str, Any], observation: Observation) -> dict[str, Any]:
        """Recheck and consume an offer, ignoring untrusted supplied subgoals."""
        if self.active:
            raise GoalValidationError('plan already active; wait for whole-plan completion')
        selection = envelope.get('goal_selection')
        if not isinstance(selection, dict):
            raise GoalValidationError('goal_selection must contain snapshot_id and goal_id')
        if self._offer is None or selection.get('snapshot_id') != self._offer['snapshot_id']:
            raise GoalValidationError('unknown or consumed snapshot; request the latest offer')
        if self._stale(observation):
            self.invalidate()
            raise GoalValidationError('stale offer: episode, control, collection or position changed')
        goal_id = selection.get('goal_id')
        offered = next((candidate for candidate in self._offer['candidates']
                        if candidate['goal_id'] == goal_id), None)
        if offered is None or not offered['feasible']:
            raise GoalValidationError('unknown or infeasible goal_id in this offer')
        source = envelope.get('source')
        if source not in ('llm', 'fallback', 'budget'):
            raise GoalValidationError('selection source must be llm, fallback or budget')
        plan_id = envelope.get('plan_id')
        if not isinstance(plan_id, str) or not plan_id.strip() or len(plan_id) > 200:
            raise GoalValidationError('plan_id must be a non-empty string of at most 200 characters')
        # Time, battery, terrain and map may have changed during an API request.
        # Recompute feasibility instead of equating clock advance with staleness.
        current = self._evaluate(observation, self._targets)
        candidate = next(item for item in current['candidates'] if item['goal_id'] == goal_id)
        if not candidate['feasible']:
            self.invalidate()
            raise GoalValidationError(f'goal no longer feasible: {candidate["reason"]}')
        compiled = {'plan_id': plan_id, 'source': source,
                    'subgoals': candidate_steps(candidate),
                    'goal_selection': deepcopy(selection)}
        self._active = {**deepcopy(candidate), 'plan_id': plan_id,
                        'collected': observation.collected,
                        'subgoal_count': len(compiled['subgoals'])}
        self.invalidate()
        return compiled

    def complete(self, status: dict[str, Any]) -> bool:
        """Only a terminal whole-plan outcome records an attempted goal."""
        active = self._active
        if active is None or status.get('plan_id') != active['plan_id']:
            return False
        state = status.get('state')
        terminal = state in ('failed', 'preempted') or (
            state == 'done' and status.get('index') == active['subgoal_count'] - 1)
        if not terminal:
            return False
        opportunistic_stop = (
            state == 'failed'
            and status.get('type') == 'goto'
            and status.get('reason') == 'sample signal nearby'
        )
        if state != 'preempted' and active['kind'] == 'search' and not opportunistic_stop:
            self._attempted.add(active['goal_id'])
            self._history.append({
                'goal_id': active['goal_id'], 'x': active['x'], 'y': active['y'],
                'collected': active['collected'], 'state': state,
                'type': status.get('type'), 'reason': str(status.get('reason', '')),
            })
        self._active = None
        return True
