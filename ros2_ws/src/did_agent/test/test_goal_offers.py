"""Live planner regressions: immutable offers, fresh budgets and lifecycle."""

from dataclasses import replace

import numpy as np
import pytest

from did_agent.costmap import CostMap
from did_agent.experimental_goals import GoalValidationError, Observation
from did_agent.goal_offers import GoalOffers
from did_agent.grid import GridMap


@pytest.fixture
def setup():
    grid = GridMap(0.5, 0.0, 0.0, np.zeros((8, 8), dtype=bool),
                   np.zeros((8, 8), dtype=bool))
    costmap = CostMap(grid, robot_radius=0.0, inflation_radius=0.5, wall_weight=0.0)
    observation = Observation('episode-1', 0, (0.25, 0.25), 100.0, samples_total=3)
    return costmap, observation, GoalOffers(costmap, observation.pose)


def request(offer, goal_id=None, **extra):
    return {'plan_id': 'decision-1', 'source': 'llm',
            'goal_selection': {'snapshot_id': offer['snapshot_id'],
                               'goal_id': goal_id or offer['candidates'][0]['goal_id']},
            'subgoals': [{'type': 'goto', 'x': 9, 'y': 9}], **extra}


def test_offer_survives_slow_model_and_small_idle_drift(setup):
    _, observation, manager = setup
    offer = manager.build(observation)
    later = replace(observation, time=180, pose=(0.28, 0.26), battery=99)
    assert manager.build(later) == offer
    plan = manager.accept(request(offer), later)
    assert plan['subgoals'] == offer['candidates'][0]['subgoals']


def test_client_cannot_change_compiled_coordinates_or_energy(setup):
    _, observation, manager = setup
    offer = manager.build(observation)
    expected = offer['candidates'][0]['subgoals']
    offer['candidates'][0]['x'] = 9
    offer['candidates'][0]['required_battery'] = -100
    plan = manager.accept(request(offer), observation)
    assert plan['subgoals'] == expected
    assert plan['subgoals'][0]['x'] != 9
    assert manager.offer is None


@pytest.mark.parametrize('change', [
    {'episode_id': 'next'}, {'revision': 1}, {'collected': 1},
    {'samples_total': 4}, {'pose': (0.55, 0.25)},
])
def test_semantic_changes_reject_delayed_decision(setup, change):
    _, observation, manager = setup
    offer = manager.build(observation)
    with pytest.raises(GoalValidationError, match='stale offer'):
        manager.accept(request(offer), replace(observation, **change))
    assert manager.offer is None


def test_manual_invalidation_revokes_even_identical_observation(setup):
    _, observation, manager = setup
    old = manager.build(observation)
    manager.invalidate(interrupt=True)
    new = manager.build(observation)
    assert old['snapshot_id'] != new['snapshot_id']
    with pytest.raises(GoalValidationError, match='unknown or consumed'):
        manager.accept(request(old), observation)


@pytest.mark.parametrize('change', ['battery', 'terrain', 'blocked'])
def test_acceptance_rechecks_current_feasibility(setup, change):
    costmap, observation, manager = setup
    offer = manager.build(observation, sensor=0.6)
    goal = offer['candidates'][0]
    assert goal['feasible'] and goal['kind'] == 'search'
    if change == 'battery':
        observation = replace(observation, battery=1)
    elif change == 'terrain':
        costmap.terrain[:] = 100
    else:
        costmap.blocked[costmap.world_to_cell(goal['x'], goal['y'])] = True
    with pytest.raises(GoalValidationError, match='no longer feasible'):
        manager.accept(request(offer), observation)
    assert manager.offer is None


def test_irrelevant_map_change_is_revalidated_without_blanket_rejection(setup):
    costmap, observation, manager = setup
    offer = manager.build(observation)
    costmap.terrain[-1, -1] = 2
    assert manager.accept(request(offer), replace(observation, time=120))


def test_active_plan_rejects_late_response_and_does_not_mark_planned_area(setup):
    _, observation, manager = setup
    offer = manager.build(observation, sensor=0.6)
    plan = manager.accept(request(offer), observation)
    assert manager.build(observation) is None
    with pytest.raises(GoalValidationError, match='already active'):
        manager.accept(request(offer), observation)
    assert not manager.complete({'plan_id': plan['plan_id'], 'index': 0,
                                 'type': 'goto', 'state': 'done'})
    assert not manager.complete({'plan_id': plan['plan_id'], 'index': 1,
                                 'type': 'search_around', 'state': 'done'})
    assert manager.active and manager.build(observation) is None
    assert manager.complete({'plan_id': plan['plan_id'], 'index': 2,
                             'type': 'collect', 'state': 'done'})
    following = manager.build(replace(observation, collected=1), sensor=0.5)
    assert plan['goal_selection']['goal_id'] in following['observations']['attempted_goal_ids']
    assert following['candidates'][0]['goal_id'].endswith('_n1')


def test_local_failure_uses_next_exploration_instead_of_repeating_search(setup):
    _, observation, manager = setup
    offer = manager.build(observation, sensor=0.6)
    goal_id = offer['candidates'][0]['goal_id']
    assert goal_id.startswith('local_')
    manager.accept(request(offer), observation)
    manager.complete({'plan_id': 'decision-1', 'index': 1, 'type': 'search_around',
                      'state': 'failed', 'reason': 'no signal'})
    following = manager.build(replace(observation, pose=(0.3, 0.3)), sensor=0.6)
    assert all(not item['goal_id'].startswith('local_') for item in following['candidates'])
    assert following['observations']['recent_attempts'][0]['reason'] == 'no signal'


def test_preempted_plan_does_not_mark_unexecuted_search_attempted(setup):
    _, observation, manager = setup
    offer = manager.build(observation)
    manager.accept(request(offer), observation)
    manager.complete({'plan_id': 'decision-1', 'index': 0, 'state': 'preempted'})
    following = manager.build(observation)
    assert following['observations']['attempted_goal_ids'] == []
    assert following['candidates'][0]['goal_id'] == offer['candidates'][0]['goal_id']
    with pytest.raises(GoalValidationError, match='unknown or consumed'):
        manager.accept(request(offer), observation)


def test_signal_interrupted_transit_is_reoffered_from_fresh_pose(setup):
    _, observation, manager = setup
    offer = manager.build(observation)
    broad = next(candidate for candidate in offer['candidates']
                 if candidate['kind'] == 'search' and not candidate['goal_id'].startswith('local_'))
    manager.accept(request(offer, broad['goal_id']), observation)
    assert manager.complete({
        'plan_id': 'decision-1', 'index': 0, 'type': 'goto',
        'state': 'failed', 'reason': 'sample signal nearby',
    })

    stopped = replace(observation, pose=(0.45, 0.25))
    following = manager.build(stopped, sensor=0.7)
    assert broad['goal_id'] not in following['observations']['attempted_goal_ids']
    assert following['observations']['recent_attempts'] == []
    assert following['candidates'][0]['goal_id'].startswith('local_')


def test_reserve_breach_offers_best_effort_home_and_no_search(setup):
    _, observation, manager = setup
    observation = replace(observation, battery=1, pose=(1.25, 0.25))
    offer = manager.build(observation, sensor=0.8)
    feasible = [candidate for candidate in offer['candidates'] if candidate['feasible']]
    assert len(feasible) == 1
    assert feasible[0]['goal_id'] == 'home' and feasible[0]['emergency']
    assert 'no guarantee' in feasible[0]['reason']
    assert manager.accept(request(offer, 'home', source='budget'), observation)['subgoals'] == [
        {'type': 'return_to_base'}]


def test_blocked_start_can_recover_but_blocked_targets_are_never_offered(setup):
    # Recovery must fit strictly inside the navigator's 0.5 m snap radius.
    grid = GridMap(0.25, 0.0, 0.0, np.zeros((16, 16), dtype=bool),
                   np.zeros((16, 16), dtype=bool))
    costmap = CostMap(grid, robot_radius=0.0, inflation_radius=0.5, wall_weight=0.0)
    observation = Observation('episode-1', 0, (0.125, 0.125), 100.0, samples_total=3)
    manager = GoalOffers(costmap, (1.125, 0.125))
    costmap.blocked[0, 0] = True
    # Keep home free while the current robot cell has just been blocked.
    offer = manager.build(observation)
    assert any(candidate['feasible'] for candidate in offer['candidates'])
    assert all(costmap.is_free(*costmap.world_to_cell(candidate['x'], candidate['y']))
               for candidate in offer['candidates'])


def test_model_cannot_finish_early_while_search_budget_remains(setup):
    _, observation, manager = setup
    offer = manager.build(observation)
    assert any(c['kind'] == 'search' and c['feasible'] for c in offer['candidates'])
    with pytest.raises(GoalValidationError, match='infeasible'):
        manager.accept(request(offer, 'home'), observation)


def test_map_changes_do_not_rename_a_failed_target_into_an_immediate_retry():
    grid = GridMap(0.1, 0.0, 0.0, np.zeros((20, 20), dtype=bool),
                   np.zeros((20, 20), dtype=bool))
    costmap = CostMap(grid, robot_radius=0.0, inflation_radius=0.5, wall_weight=0.0)
    observation = Observation('episode', 0, (0.55, 0.55), 100.0, samples_total=3)
    manager = GoalOffers(costmap, observation.pose)
    offer = manager.build(observation)
    target = next(c for c in offer['candidates'] if c['kind'] == 'search')
    manager.accept(request(offer, target['goal_id']), observation)
    manager.complete({'plan_id': 'decision-1', 'index': 0, 'state': 'failed'})
    costmap.blocked[costmap.world_to_cell(target['x'], target['y'])] = True
    following = manager.build(observation)
    assert all(np.hypot(c['x'] - target['x'], c['y'] - target['y']) >= 0.45
               for c in following['candidates'] if c['kind'] == 'search')


def test_unknown_goal_and_untrusted_source_are_rejected(setup):
    _, observation, manager = setup
    offer = manager.build(observation)
    with pytest.raises(GoalValidationError, match='unknown or infeasible'):
        manager.accept(request(offer, 'made-up'), observation)
    with pytest.raises(GoalValidationError, match='source'):
        manager.accept(request(offer, source='manual'), observation)
