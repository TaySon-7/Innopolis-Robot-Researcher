"""Math/protocol regressions for the opt-in, ROS-independent experiment."""

from dataclasses import replace
import json
from math import hypot

import numpy as np
import pytest

from did_agent.costmap import CostMap
from did_agent.experimental_goals import (
    BudgetPolicy, CandidateBackend, GoalValidationError, Observation, SearchTarget,
    search_targets,
)
from did_agent.grid import GridMap
from did_agent.plan import parse_plan


@pytest.fixture
def costmap():
    grid = GridMap(0.5, 0.0, 0.0, np.zeros((8, 8), dtype=bool),
                   np.zeros((8, 8), dtype=bool))
    return CostMap(grid, robot_radius=0.0, inflation_radius=0.5, wall_weight=0.0)


@pytest.fixture
def observation():
    return Observation('episode-1', 0, (0.25, 0.25), 100.0)


def target():
    return SearchTarget('A', 1.25, 0.25, 0.3)


def test_budget_uses_route_search_and_return_reserve(costmap, observation):
    backend = CandidateBackend(costmap, observation.pose)
    batch = backend.build(observation, [target()])
    candidate = batch['candidates'][0]
    assert candidate['energy_to_goal'] == pytest.approx(1.0)
    assert candidate['energy_home'] == pytest.approx(1.0)
    assert candidate['energy_search'] == pytest.approx(3.0)
    assert candidate['required_battery'] == pytest.approx(1 + 3 + 1.4 + 8)
    assert candidate['feasible']
    assert batch['candidates'][-1]['goal_id'] == 'home'
    assert batch['objective']['weights'] == {'N': 10.0, 'R': 5.0, 'C': -2.0, 'F': -1.0, 'H': -3.0}
    assert batch['objective']['expected_score'] is None
    assert all(c['expected_score'] is None for c in batch['candidates'])
    json.dumps(batch, allow_nan=False)


def test_exact_threshold_and_insufficient_return(costmap, observation):
    backend = CandidateBackend(costmap, observation.pose)
    needed = backend.build(observation, [target()])['candidates'][0]['required_battery']
    assert backend.build(replace(observation, battery=needed), [target()])['candidates'][0]['feasible']
    assert not backend.build(replace(observation, battery=needed - 0.001), [target()])['candidates'][0]['feasible']
    batch = backend.build(replace(observation, battery=1.0), [target()])
    assert not any(c['feasible'] for c in batch['candidates'])
    with pytest.raises(GoalValidationError, match='insufficient battery'):
        backend.accept(batch['snapshot_id'], 'home', replace(observation, battery=1.0))


def test_energy_units_scale_all_legs_but_not_reserve(costmap, observation):
    backend = CandidateBackend(costmap, observation.pose, BudgetPolicy(energy_per_meter=2.0))
    candidate = backend.build(observation, [target()])['candidates'][0]
    assert candidate['required_battery'] == pytest.approx(2 * (1 + 3 + 1.4) + 8)


def test_home_budget_uses_current_pose_to_base(costmap, observation):
    backend = CandidateBackend(costmap, observation.pose)
    current = replace(observation, pose=(1.25, 0.25))
    home = backend.build(current, [])['candidates'][0]
    assert home['energy_to_goal'] == pytest.approx(1.0)
    assert home['energy_search'] == home['energy_home'] == 0
    assert home['required_battery'] == pytest.approx(1.4 + 8)


def test_expensive_observed_terrain_changes_candidate_cost(costmap, observation):
    backend = CandidateBackend(costmap, observation.pose)
    targets = [SearchTarget('A', 1.25, 0.25, 0.3), SearchTarget('B', 0.25, 1.25, 0.3)]
    baseline = backend.build(observation, targets)['candidates']
    assert baseline[0]['required_battery'] == pytest.approx(baseline[1]['required_battery'])
    costmap.terrain[0, 2] = 4
    costmap.last_seen[0, 2] = 0
    candidates = backend.build(observation, targets)['candidates']
    assert candidates[0]['required_battery'] > candidates[1]['required_battery']
    assert candidates[0]['energy_search'] == pytest.approx(3 * (1 + 3 * 1.5))


def test_wall_penalty_is_not_battery_consumption(costmap, observation):
    backend = CandidateBackend(costmap, observation.pose)
    ordinary = backend.build(observation, [target()])['candidates'][0]
    costmap.wall_cost[:] = 50.0
    inflated = backend.build(observation, [target()])['candidates'][0]
    assert ordinary['required_battery'] == pytest.approx(inflated['required_battery'])


def test_blocked_target_is_not_snapped_to_another_cell(costmap, observation):
    costmap.blocked[0, 2] = True
    batch = CandidateBackend(costmap, observation.pose).build(observation, [target()])
    candidate = batch['candidates'][0]
    assert not candidate['reachable'] and not candidate['feasible']
    assert candidate['energy_to_goal'] is None
    assert candidate['required_battery'] is None


def test_disconnected_target_is_rejected(costmap, observation):
    costmap.blocked[:, 1] = True
    backend = CandidateBackend(costmap, observation.pose)
    batch = backend.build(observation, [target()])
    assert not batch['candidates'][0]['reachable']
    with pytest.raises(GoalValidationError, match='no free route'):
        backend.accept(batch['snapshot_id'], 'A', observation)


def test_accept_uses_private_targets_and_consumes_snapshot(costmap, observation):
    backend = CandidateBackend(costmap, observation.pose)
    batch = backend.build(observation, [target()])
    batch['candidates'][0]['x'] = 3.0
    batch['candidates'][0]['energy_to_goal'] = -100
    plan = backend.accept(batch['snapshot_id'], 'A', observation)
    parsed = parse_plan(json.dumps(plan))
    assert [s.type for s in parsed.subgoals] == ['goto', 'search_around', 'collect']
    assert parsed.subgoals[0].x == target().x
    with pytest.raises(GoalValidationError, match='consumed snapshot'):
        backend.accept(batch['snapshot_id'], 'A', observation)


@pytest.mark.parametrize('changes', [
    {'episode_id': 'episode-2'}, {'revision': 1}, {'battery': 90.0},
    {'pose': (0.75, 0.25)}, {'time': 0.01}, {'collected': 1},
])
def test_stale_observation_is_rejected(costmap, observation, changes):
    backend = CandidateBackend(costmap, observation.pose)
    batch = backend.build(observation, [target()])
    with pytest.raises(GoalValidationError, match='stale snapshot'):
        backend.accept(batch['snapshot_id'], 'A', replace(observation, **changes))


@pytest.mark.parametrize('layer', ['terrain', 'wall_cost', 'last_seen', 'blocked'])
def test_map_changes_without_version_bump_are_rejected(costmap, observation, layer):
    backend = CandidateBackend(costmap, observation.pose)
    batch = backend.build(observation, [target()])
    before_version = costmap.version
    array = getattr(costmap, layer)
    array[5, 5] = True if layer == 'blocked' else float(array[5, 5]) + 0.1
    # Float32 last_seen=-1e9 cannot represent a 0.1 increment.
    if layer == 'last_seen':
        array[5, 5] = 0.0
    assert costmap.version == before_version
    with pytest.raises(GoalValidationError, match='stale snapshot'):
        backend.accept(batch['snapshot_id'], 'A', observation)


def test_unknown_goal_and_unknown_snapshot_are_rejected(costmap, observation):
    backend = CandidateBackend(costmap, observation.pose)
    batch = backend.build(observation, [target()])
    with pytest.raises(GoalValidationError, match='unknown goal_id'):
        backend.accept(batch['snapshot_id'], 'invented', observation)
    with pytest.raises(GoalValidationError, match='unknown or consumed snapshot'):
        backend.accept('invented', 'A', observation)


@pytest.mark.parametrize('changes', [
    {'battery': float('nan')}, {'battery': float('inf')}, {'battery': True},
    {'battery': -1}, {'revision': True}, {'revision': -1},
    {'time': float('nan')}, {'pose': (float('nan'), 0.0)}, {'pose': (9.0, 9.0)},
    {'collected': 2}, {'samples_total': -1}, {'episode_id': ''},
])
def test_invalid_observation_is_rejected(costmap, observation, changes):
    with pytest.raises(GoalValidationError):
        CandidateBackend(costmap, observation.pose).build(replace(observation, **changes), [target()])


@pytest.mark.parametrize('targets', [
    [SearchTarget('home', 1.25, 0.25)], [target(), target()],
    [SearchTarget('A', True, 0.25)], [SearchTarget('A', 1.25, 0.25, float('inf'))],
    [SearchTarget('A', 1.25, 0.25, 0.01)], [SearchTarget('A', 5.0, 5.0)],
])
def test_invalid_targets_are_rejected(costmap, observation, targets):
    with pytest.raises(GoalValidationError):
        CandidateBackend(costmap, observation.pose).build(observation, targets)


@pytest.mark.parametrize('policy', [
    BudgetPolicy(energy_per_meter=0), BudgetPolicy(search_distance=0),
    BudgetPolicy(reserve=-1), BudgetPolicy(return_factor=0.5),
    BudgetPolicy(pessimism=float('nan')), BudgetPolicy(reserve=True),
])
def test_invalid_policy_is_rejected(costmap, observation, policy):
    with pytest.raises(GoalValidationError):
        CandidateBackend(costmap, observation.pose, policy)


def test_overflowing_search_estimate_is_rejected_even_for_blocked_target(costmap, observation):
    costmap.blocked[0, 2] = True
    backend = CandidateBackend(costmap, observation.pose, BudgetPolicy(energy_per_meter=1e308))
    with pytest.raises(GoalValidationError, match='search energy estimate is not finite'):
        backend.build(observation, [target()])


def test_finished_collection_only_allows_home(costmap, observation):
    observation = replace(observation, collected=1)
    backend = CandidateBackend(costmap, observation.pose)
    batch = backend.build(observation, [target()])
    assert not batch['candidates'][0]['feasible']
    assert backend.accept(batch['snapshot_id'], 'home', observation)['subgoals'] == [
        {'type': 'return_to_base'}]


def test_identical_observations_and_maps_make_identical_batches(costmap, observation):
    first = CandidateBackend(costmap, observation.pose).build(observation, [target()])
    second = CandidateBackend(costmap, observation.pose).build(observation, [target()])
    assert first == second
    # No input object exposes scenario samples, future events or judge state.
    assert set(observation.__dataclass_fields__) == {
        'episode_id', 'revision', 'pose', 'battery', 'time', 'collected', 'samples_total',
    }


def test_target_helper_uses_only_free_cells_with_stable_ids(costmap):
    costmap.blocked[0, :] = True
    first = search_targets(costmap)
    assert first == search_targets(costmap)
    assert len(first) == 5
    assert all(costmap.is_free(*costmap.world_to_cell(t.x, t.y)) for t in first)
    assert search_targets(costmap, limit=0) == []
    later = search_targets(costmap, exclude=[first[0].goal_id])
    assert first[0].goal_id not in {t.goal_id for t in later}
    assert all(hypot(t.x - first[0].x, t.y - first[0].y) >= 1.3 for t in later)
    for candidate in later:
        row, col = costmap.world_to_cell(candidate.x, candidate.y)
        assert candidate.goal_id == f'cell_{row}_{col}'


def test_target_limit_is_applied_after_connectivity(costmap):
    costmap.blocked[3, :] = True
    start = costmap.cell_to_world(4, 0)
    # The lower pocket appears first in row order and previously filled the cap.
    assert all(costmap.world_to_cell(t.x, t.y)[0] < 3
               for t in search_targets(costmap, spacing=0.1, limit=4))
    targets = search_targets(costmap, spacing=0.1, limit=4, start=start)
    assert len(targets) == 4
    assert all(costmap.world_to_cell(t.x, t.y)[0] >= 4 for t in targets)
    batch = CandidateBackend(costmap, start).build(Observation('episode', 0, start, 100), targets)
    assert all(candidate['reachable'] for candidate in batch['candidates'])


def test_connectivity_does_not_cross_blocked_diagonal_corners(costmap):
    costmap.blocked[:] = True
    costmap.blocked[0, 0] = False
    costmap.blocked[1, 1:4] = False
    start = costmap.cell_to_world(0, 0)
    targets = search_targets(costmap, spacing=0.1, limit=4, start=start)
    assert [t.goal_id for t in targets] == ['cell_0_0']
    costmap.blocked[0, 1] = False  # Open an orthogonal route into the corridor.
    targets = search_targets(costmap, spacing=0.1, limit=5, start=start)
    assert {t.goal_id for t in targets} == {
        'cell_0_0', 'cell_0_1', 'cell_1_1', 'cell_1_2', 'cell_1_3',
    }


def test_reachable_exclusions_keep_stable_cell_ids_and_spacing(costmap):
    costmap.blocked[3, :] = True
    start = costmap.cell_to_world(4, 0)
    original = search_targets(costmap, start=start, limit=4)
    excluded = original[0]
    later = search_targets(costmap, start=start, limit=4, exclude=[excluded.goal_id])
    assert later
    assert later == search_targets(costmap, start=start, limit=4, exclude=[excluded.goal_id])
    for candidate in later:
        row, col = costmap.world_to_cell(candidate.x, candidate.y)
        assert row >= 4
        assert candidate.goal_id == f'cell_{row}_{col}'
        assert hypot(candidate.x - excluded.x, candidate.y - excluded.y) >= 1.3


@pytest.mark.parametrize('start', [(True, 0), (float('nan'), 0), (4, 4), (0,)])
def test_invalid_target_generation_start_is_rejected(costmap, start):
    with pytest.raises(GoalValidationError):
        search_targets(costmap, start=start)


def test_blocked_start_produces_no_reachable_targets(costmap):
    costmap.blocked[0, 0] = True
    assert search_targets(costmap, start=(0.25, 0.25)) == []
