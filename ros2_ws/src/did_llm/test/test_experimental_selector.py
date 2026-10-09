"""Offline regression tests for the optional ID-only goal-selection experiment."""

from copy import deepcopy
from dataclasses import replace
import json

import numpy as np
import pytest

from did_agent.costmap import CostMap
from did_agent.experimental_goals import (
    CandidateBackend, GoalValidationError, Observation, SearchTarget,
)
from did_agent.grid import GridMap
from did_llm.experimental_selector import (
    BudgetStub, ChoiceRejected, DecisionSession, Selection, SessionError,
    choice_response_format, parse_choice, select_goal,
)
from did_llm.llm_client import LLMUnavailable


class ScriptedClient:
    """Only returns local values; records the actual model-facing messages."""

    def __init__(self, *answers):
        self.answers = iter(answers)
        self.calls = []

    def complete_json(self, system, user, **kwargs):
        self.calls.append({'system': system, 'request': json.loads(user), **kwargs})
        answer = next(self.answers)
        if isinstance(answer, Exception):
            raise answer
        return deepcopy(answer)


@pytest.fixture
def experiment():
    free = np.zeros((8, 8), dtype=bool)
    costmap = CostMap(GridMap(0.5, -2.0, -2.0, free, free.copy()))
    costmap.blocked[costmap.world_to_cell(0.75, 0.75)] = True
    backend = CandidateBackend(costmap, base=(-1.25, -1.25))
    observation = Observation('episode-1', 0, (-1.25, -1.25), 100.0)
    targets = [SearchTarget('A', 0.25, -0.75),
               SearchTarget('B', 1.25, 1.25),
               SearchTarget('blocked', 0.75, 0.75)]
    return costmap, backend, observation, targets


def activate(experiment):
    _, backend, observation, targets = experiment
    session = DecisionSession(backend)
    ticket, batch = session.prepare(observation, targets)
    selection = select_goal(batch, ScriptedClient({'goal_id': 'A', 'reason': 'Search A.'}))
    active = session.commit(ticket, selection, observation)
    return session, active


def status(active, index, state='done', **fields):
    return {'plan_id': active['plan']['plan_id'], 'index': index, 'state': state, **fields}


@pytest.mark.parametrize('answer', [
    None,
    [],
    {'goal_id': 'A'},
    {'goal_id': 'A', 'reason': 'Search.', 'x': 0},
    {'goal_id': True, 'reason': 'Search.'},
    {'goal_id': 1, 'reason': 'Search.'},
    {'goal_id': 'invented', 'reason': 'Search.'},
    {'goal_id': 'blocked', 'reason': 'Ignore the route.'},
    {'goal_id': 'A', 'reason': None},
    {'goal_id': 'A', 'reason': ''},
    {'goal_id': 'A', 'reason': ' \n '},
    {'goal_id': 'A', 'reason': 'x' * 501},
])
def test_model_cannot_expand_the_decision_contract(experiment, answer):
    _, backend, observation, targets = experiment
    batch = backend.build(observation, targets)
    with pytest.raises(ChoiceRejected):
        parse_choice(answer, batch)


def test_schema_only_offers_ids_that_the_backend_can_execute(experiment):
    _, backend, observation, targets = experiment
    batch = backend.build(observation, targets)
    schema = choice_response_format(batch)['json_schema']
    assert schema['strict'] is True
    assert schema['schema']['additionalProperties'] is False
    assert set(schema['schema']['required']) == {'goal_id', 'reason'}
    assert set(schema['schema']['properties']['goal_id']['enum']) == {'A', 'B', 'home'}


def test_repair_reports_validation_error_and_preserves_the_offer(experiment):
    _, backend, observation, targets = experiment
    batch = backend.build(observation, targets)
    client = ScriptedClient({'goal_id': 'blocked', 'reason': 'Search.'},
                            {'goal_id': 'A', 'reason': 'A has a feasible route.'})
    selected = select_goal(batch, client)
    assert selected.source == 'llm'
    assert selected.choice['goal_id'] == 'A'
    assert len(client.calls) == 2
    assert client.calls[0]['request']['validation_errors'] == []
    assert 'feasible' in client.calls[1]['request']['validation_errors'][0]
    assert client.calls[1]['request']['offer'] == client.calls[0]['request']['offer']
    assert all(call['response_format'] == choice_response_format(batch) for call in client.calls)


def test_second_invalid_answer_ends_repair_and_labels_the_fallback(experiment):
    _, backend, observation, targets = experiment
    batch = backend.build(observation, targets)
    client = ScriptedClient({'goal_id': 'invented', 'reason': 'Try.'},
                            {'goal_id': 'A', 'reason': 'Try.', 'velocity': 1})
    selected = select_goal(batch, client)
    assert len(client.calls) == 2
    assert selected.source == 'fallback'
    assert len(selected.errors) == 2
    assert parse_choice(selected.choice, batch) == selected.choice
    assert selected.choice['goal_id'] != 'blocked'


def test_provider_failure_does_not_leak_its_body_into_the_result(experiment):
    _, backend, observation, targets = experiment
    client = ScriptedClient(LLMUnavailable('provider says api_key=SECRET_TOKEN'))
    selected = select_goal(backend.build(observation, targets), client)
    assert selected.source == 'fallback'
    assert len(client.calls) == 1
    assert selected.errors
    assert 'SECRET_TOKEN' not in repr(selected)


def test_offline_stub_is_explicitly_distinguished_from_llm(experiment):
    _, backend, observation, targets = experiment
    selected = select_goal(backend.build(observation, targets), BudgetStub())
    assert selected.source == 'stub'


def test_no_feasible_goal_makes_no_model_call_and_no_motion_plan(experiment):
    _, backend, observation, targets = experiment
    empty_battery = replace(observation, battery=0.0)
    batch = backend.build(empty_battery, targets)
    client = ScriptedClient()
    with pytest.raises(ChoiceRejected, match='No feasible'):
        select_goal(batch, client)
    assert client.calls == []
    for goal_id in ('A', 'B', 'home'):
        with pytest.raises(GoalValidationError):
            backend.accept(batch['snapshot_id'], goal_id, empty_battery)


def test_empty_offer_does_not_leave_session_permanently_busy(experiment):
    _, backend, observation, targets = experiment
    session = DecisionSession(backend)
    with pytest.raises(ChoiceRejected, match='No feasible'):
        session.prepare(replace(observation, battery=0.0), targets)
    assert not session.busy
    session.prepare(observation, targets)
    assert session.busy


def test_real_client_with_unparseable_stub_response_falls_back(experiment, monkeypatch):
    from did_llm.llm_client import LLMClient, LLMConfig

    _, backend, observation, targets = experiment
    client = LLMClient(LLMConfig(base_url='https://example.test', api_key='fake',
                                model='stub', min_interval_sec=0))
    monkeypatch.setattr(client, '_post', lambda *_args, **_kwargs: 'not JSON')
    selection = select_goal(backend.build(observation, targets), client)
    assert selection.source == 'fallback'
    assert client._budget.calls_made == 1


def test_mutating_the_model_offer_cannot_change_executable_coordinates(experiment):
    _, backend, observation, targets = experiment
    session = DecisionSession(backend)
    ticket, batch = session.prepare(observation, targets)
    original = next(c for c in batch['candidates'] if c['goal_id'] == 'A')
    original_xy = original['x'], original['y']
    original['x'], original['y'], original['radius'] = 99.0, 99.0, 99.0
    selection = Selection(batch['snapshot_id'], {'goal_id': 'A', 'reason': 'Search.'}, 'llm')
    active = session.commit(ticket, selection, observation)
    goto, search, collect = active['plan']['subgoals']
    assert (goto['x'], goto['y']) == original_xy
    assert (search['x'], search['y']) == original_xy
    assert search['radius'] == targets[0].radius
    assert collect == {'type': 'collect'}
    # The returned execution envelope is also disposable to callers.
    active['plan']['subgoals'].clear()
    assert session.on_status({'plan_id': active['plan']['plan_id'], 'index': 0, 'state': 'done'}) is False
    assert session.busy


@pytest.mark.parametrize('changed', ['battery', 'map', 'episode', 'pose', 'revision'])
def test_changed_robot_or_map_state_rejects_an_in_flight_model_answer(experiment, changed):
    costmap, backend, observation, targets = experiment
    session = DecisionSession(backend)
    ticket, batch = session.prepare(observation, targets)
    selection = select_goal(batch, BudgetStub())
    current = observation
    if changed == 'battery':
        current = replace(observation, battery=observation.battery - 1.0)
    elif changed == 'map':
        # Small terrain updates need not increment CostMap.version.
        costmap.terrain[0, 0] = 1.1
    elif changed == 'episode':
        current = replace(observation, episode_id='episode-2')
    elif changed == 'pose':
        current = replace(observation, pose=(-0.75, -1.25))
    else:
        current = replace(observation, revision=observation.revision + 1)
    with pytest.raises(GoalValidationError, match='stale'):
        session.commit(ticket, selection, current)
    assert session.busy is False
    assert not session.attempted_goal_ids
    assert session.outcomes == []


def test_search_cannot_be_replaced_after_only_goto_or_search_completion(experiment):
    _, _, observation, targets = experiment
    session, active = activate(experiment)
    assert not session.attempted_goal_ids
    assert session.outcomes == []
    for index in (0, 1):
        assert session.on_status(status(active, index)) is False
        assert session.busy
        assert not session.attempted_goal_ids
        with pytest.raises(SessionError):
            session.prepare(observation, targets)
    assert session.on_status(status(active, 2)) is True
    assert session.busy is False
    assert session.attempted_goal_ids == {'A'}
    assert session.outcomes[0]['state'] == 'done'


def test_stale_duplicate_and_out_of_order_statuses_do_not_finish_a_plan(experiment):
    session, active = activate(experiment)
    assert session.on_status(status(active, 2, plan_id='old-plan')) is False
    with pytest.raises(SessionError, match='Out-of-order'):
        session.on_status(status(active, 2))
    assert session.on_status(status(active, 0, 'running')) is False
    assert session.on_status(status(active, 0)) is False
    assert session.on_status(status(active, 0)) is False
    assert session.on_status(status(active, 1)) is False
    assert session.on_status(status(active, 2)) is True
    assert session.on_status(status(active, 2)) is False
    assert len(session.outcomes) == 1


@pytest.mark.parametrize('index', [True, 0.0, '0', None, -1])
def test_status_indices_must_not_coerce_to_a_valid_subgoal(experiment, index):
    session, active = activate(experiment)
    with pytest.raises(SessionError, match='integer'):
        session.on_status(status(active, index))
    assert session.busy
    assert session.outcomes == []


def test_failed_goal_is_not_immediately_offered_again(experiment):
    _, _, observation, targets = experiment
    session, active = activate(experiment)
    assert session.on_status(status(active, 0, 'failed', reason='No route')) is True
    assert session.attempted_goal_ids == {'A'}
    _, next_offer = session.prepare(observation, targets)
    assert 'A' not in {c['goal_id'] for c in next_offer['candidates']}
    assert session.outcomes[0]['reason'] == 'No route'


def test_preemption_does_not_claim_a_target_was_searched(experiment):
    _, _, observation, targets = experiment
    session, active = activate(experiment)
    session.on_status(status(active, 0))
    assert session.on_status(status(active, 1, 'preempted')) is True
    assert not session.attempted_goal_ids
    _, next_offer = session.prepare(observation, targets)
    assert 'A' in {c['goal_id'] for c in next_offer['candidates']}
    assert session.outcomes[0]['state'] == 'preempted'


@pytest.mark.parametrize('invalidate', ['reset', 'cancel'])
def test_request_identity_rejects_old_answer_even_for_identical_snapshot(experiment, invalidate):
    _, backend, observation, targets = experiment
    session = DecisionSession(backend)
    old_ticket, old_batch = session.prepare(observation, targets)
    old_answer = select_goal(old_batch, BudgetStub())
    if invalidate == 'reset':
        session.reset(observation.episode_id)
    else:
        session.cancel_pending()
    new_ticket, new_batch = session.prepare(observation, targets)
    assert old_batch['snapshot_id'] == new_batch['snapshot_id']
    with pytest.raises(SessionError, match='Stale'):
        session.commit(old_ticket, old_answer, observation)
    assert session.busy
    active = session.commit(new_ticket, select_goal(new_batch, BudgetStub()), observation)
    assert active['plan']['subgoals']


def test_episode_reset_clears_outcomes_and_attempted_targets(experiment):
    _, _, observation, targets = experiment
    session, active = activate(experiment)
    session.on_status(status(active, 0, 'failed'))
    next_episode = replace(observation, episode_id='episode-2')
    with pytest.raises(SessionError, match='reset'):
        session.prepare(next_episode, targets)
    session.reset(next_episode.episode_id)
    assert not session.busy
    assert not session.attempted_goal_ids
    assert session.outcomes == []
    _, batch = session.prepare(next_episode, targets)
    assert 'A' in {c['goal_id'] for c in batch['candidates']}


def test_completed_home_plan_does_not_exclude_home_from_future_offers(experiment):
    _, backend, observation, targets = experiment
    session = DecisionSession(backend)
    ticket, batch = session.prepare(observation, targets)
    selected = select_goal(batch, ScriptedClient({'goal_id': 'home', 'reason': 'Return.'}))
    active = session.commit(ticket, selected, observation)
    assert active['plan']['subgoals'] == [{'type': 'return_to_base'}]
    assert session.on_status(status(active, 0)) is True
    assert not session.attempted_goal_ids


def test_default_demo_exercises_protocol_without_loading_a_network_client(monkeypatch):
    from did_llm.experimental_demo import run_demo

    def forbidden(*_args, **_kwargs):
        pytest.fail('The default demo must not construct a model client or access the network.')

    monkeypatch.setattr('did_llm.experimental_demo.LLMClient', forbidden)
    monkeypatch.setattr('did_llm.llm_client.LLMClient._post', forbidden)
    monkeypatch.setattr('did_llm.llm_client.load_api_key', forbidden)
    monkeypatch.setattr('socket.create_connection', forbidden)
    result = run_demo()
    assert result['source'] == 'stub'
    assert result['active_plan_blocks_replanning'] is True
    assert result['synthetic_status_checks'][0] == {
        'step': 'goto', 'whole_plan_finished': False,
    }
    assert result['synthetic_status_checks'][-1]['whole_plan_finished'] is True
    assert result['changed_battery_rejects_old_answer'] is True
    assert result['low_battery_plan']['subgoals'] == [{'type': 'return_to_base'}]
