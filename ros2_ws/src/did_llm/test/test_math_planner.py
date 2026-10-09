"""Live lifecycle regressions without ROS, credentials, sleeps or network calls."""

from copy import deepcopy
import json
from threading import Event
from types import SimpleNamespace

import pytest

from did_llm.experimental_selector import BudgetStub
from did_llm.llm_client import LLMUnavailable
from did_llm.math_planner import MathPlanner


def offer(token='offer-1', episode=1):
    return {
        'snapshot_id': token, 'episode_id': episode, 'revision': 0, 'battery': 75.0,
        'objective': {'formula': '10*N+5*R-2*C-F-3*H'},
        'budget': {'reserve': 8.0, 'return_factor': 1.4},
        'observations': {'collected': 0, 'samples_total': 3,
                         'sensor': {'value': 0.2, 'noise_estimate': 0.01},
                         'pose': {'x': 0, 'y': 0}, 'attempted_goal_ids': []},
        'candidates': [
            {'goal_id': 'A', 'kind': 'search', 'feasible': True,
             'energy_to_goal': 2.0, 'energy_search': 3.0, 'energy_home': 4.0,
             'required_battery': 18.6, 'subgoals': [
                 {'type': 'goto', 'x': 0.5, 'y': 0.0},
                 {'type': 'search_around', 'x': 0.5, 'y': 0.0, 'radius': 0.6}]},
            {'goal_id': 'home', 'kind': 'return', 'feasible': True,
             'energy_to_goal': 1.0, 'energy_search': 0.0, 'energy_home': 0.0,
             'required_battery': 9.4, 'subgoals': [{'type': 'return_to_base'}]},
        ],
    }


class Link:
    def __init__(self):
        self.clock = 10.0
        self.age = 0.0
        self.state = {'episode_id': 1, 'finished': False, 'control_mode': 'llm',
                      'current': {'plan_id': '', 'index': 0, 'state': 'idle'},
                      'goal_offer': offer()}
        self.status = None
        self.command = None
        self.command_pending = False
        self.published = []
        self.entries = []

    def now(self):
        return self.clock

    def state_age(self):
        return self.age

    def publish_plan(self, payload):
        self.published.append(payload)

    def journal(self, kind, title, text='', status='open', **extra):
        self.entries.append({'kind': kind, 'title': title, 'text': text, 'status': status, **extra})

    def command_received(self, cmd):
        self.command, self.command_pending = cmd, True


class Client:
    def __init__(self, *answers, block=False):
        self.answers = iter(answers or ({'goal_id': 'A', 'reason': 'Observed useful search.'},))
        self.calls = []
        self.entered = Event()
        self.release = Event()
        if not block:
            self.release.set()

    def complete_json(self, system, user, **kwargs):
        self.calls.append({'system': system, 'request': json.loads(user), **kwargs})
        self.entered.set()
        if not self.release.wait(3.0):
            raise AssertionError('Test failed to release its fake model call.')
        answer = next(self.answers)
        if isinstance(answer, Exception):
            raise answer
        return deepcopy(answer)


def finish(planner):
    request = planner._request
    assert request is not None
    assert request.done.wait(3.0), 'Fake worker did not finish.'
    planner.tick()


def run_choice(link=None, client=None):
    link = link or Link()
    client = client or Client()
    planner = MathPlanner(link, client)
    planner.tick()
    finish(planner)
    return link, client, planner


def test_choice_preserves_backend_steps_and_records_mathematics():
    link, client, planner = run_choice()
    plan = link.published[0]
    assert plan['source'] == 'llm'
    assert plan['goal_selection'] == {'snapshot_id': 'offer-1', 'goal_id': 'A'}
    assert plan['subgoals'] == link.state['goal_offer']['candidates'][0]['subgoals']
    assert plan['decision'] == {
        'goal_id': 'A', 'battery': 75, 'energy_to_goal': 2, 'energy_search': 3,
        'energy_home': 4, 'required_battery': 18.6,
    }
    assert link.entries[-1]['decision'] == plan['decision']
    assert link.entries[-1]['decision_source'] == 'llm'
    assert client.calls[0]['request']['offer']['observations']['sensor']['value'] == 0.2
    link.state['goal_offer']['candidates'][0]['subgoals'][0]['x'] = 200
    assert plan['subgoals'][0]['x'] == 0.5
    assert planner.inflight == plan['plan_id']


def test_subgoal_done_does_not_end_plan_or_restart_search():
    client = Client({'goal_id': 'A', 'reason': 'First.'}, {'goal_id': 'A', 'reason': 'Next.'})
    link, _, planner = run_choice(client=client)
    plan_id = planner.inflight
    link.state['goal_offer'] = None
    link.state['current'] = {'plan_id': plan_id, 'index': 0, 'type': 'goto', 'state': 'done'}
    link.status = {'plan_id': plan_id, 'index': 0, 'state': 'done', 'plan_complete': False}
    planner.tick()
    assert planner.inflight == plan_id
    assert len(client.calls) == len(link.published) == 1
    link.status = {'plan_id': plan_id, 'index': 1, 'state': 'done', 'plan_complete': True}
    link.state['goal_offer'] = offer('offer-2')
    planner.tick()
    finish(planner)
    assert len(link.published) == 2
    assert link.published[1]['plan_id'] != plan_id


def test_idle_replacement_offer_recovers_from_dropped_terminal_status():
    client = Client({'goal_id': 'A', 'reason': 'First.'}, {'goal_id': 'A', 'reason': 'Next.'})
    link, _, planner = run_choice(client=client)
    link.state['goal_offer'] = offer('offer-2')
    link.state['current'] = {'plan_id': '', 'state': 'idle'}
    planner.tick()
    finish(planner)
    assert len(link.published) == 2


def test_pending_ack_and_timeout_never_duplicate_a_consumed_offer():
    link, client, planner = run_choice()
    for _ in range(3):
        planner.tick()
    assert len(link.published) == 1
    link.clock += 11
    planner.tick()
    assert planner.inflight is None
    planner.tick()
    assert len(link.published) == len(client.calls) == 1
    assert link.entries[-1]['status'] == 'rejected'


@pytest.mark.parametrize('mode', ['autonomous', 'manual', 'stopped', None])
def test_non_llm_modes_never_call_model(mode):
    link, client = Link(), Client()
    link.state['control_mode'] = mode
    planner = MathPlanner(link, client)
    planner.tick()
    assert client.calls == link.published == []
    assert not planner.busy


@pytest.mark.parametrize('missing', ['finished', 'stale', 'state', 'offer', 'busy'])
def test_only_fresh_idle_unfinished_state_can_request_a_plan(missing):
    link, client = Link(), Client()
    if missing == 'finished':
        link.state['finished'] = True
    elif missing == 'stale':
        link.age = 6.0
    elif missing == 'state':
        link.state = None
    elif missing == 'offer':
        link.state['goal_offer'] = None
    else:
        link.state['current']['state'] = 'running'
    planner = MathPlanner(link, client)
    planner.tick()
    assert client.calls == link.published == []


def test_new_episode_discards_answer_even_when_scenario_name_is_unchanged():
    link = Link()
    link.state['scenario'] = 'easy'
    client = Client({'goal_id': 'A', 'reason': 'Old.'}, {'goal_id': 'A', 'reason': 'New.'}, block=True)
    planner = MathPlanner(link, client)
    planner.tick()
    assert client.entered.wait(3)
    old_request = planner._request
    link.state['episode_id'] = 2
    link.state['goal_offer'] = offer('new-run', episode=2)
    planner.tick()
    assert len(client.calls) == 1
    assert link.published == []
    client.release.set()
    assert old_request.done.wait(3)
    planner.tick()
    finish(planner)
    assert len(link.published) == 1
    assert link.published[0]['goal_selection']['snapshot_id'] == 'new-run'
    assert link.published[0]['explanation'] == 'New.'


def test_stop_fences_late_answer_before_state_updates_then_explicit_llm_resumes():
    link, client = Link(), Client({'goal_id': 'A', 'reason': 'Old.'},
                                 {'goal_id': 'A', 'reason': 'Resumed.'}, block=True)
    planner = MathPlanner(link, client)
    planner.tick()
    assert client.entered.wait(3)
    old_request = planner._request
    link.command_received('stop')
    planner.tick()
    client.release.set()
    assert old_request.done.wait(3)
    planner.tick()
    assert link.published == []  # The previous state still says "llm".
    link.command_received('llm')
    link.state['goal_offer'] = offer('resume-offer')
    planner.tick()
    finish(planner)
    assert len(link.published) == 1
    assert link.published[0]['explanation'] == 'Resumed.'


def test_stop_does_not_issue_a_repair_api_call_after_first_answer_returns():
    link, client = Link(), Client({'goal_id': 'invented', 'reason': 'Old.'}, block=True)
    planner = MathPlanner(link, client)
    planner.tick()
    assert client.entered.wait(3)
    request = planner._request
    link.command_received('stop')
    planner.tick()
    client.release.set()
    assert request.done.wait(3)
    assert len(client.calls) == 1
    planner.tick()
    assert link.published == []


def test_new_episode_resumes_after_stop_even_if_stopped_state_was_not_observed():
    link, client = Link(), Client()
    planner = MathPlanner(link, client)
    link.command_received('stop')
    planner.tick()
    assert client.calls == []
    # The reset and new LLM start reached the backend between two state polls.
    link.state['episode_id'] = 2
    link.state['goal_offer'] = offer('fresh-run', episode=2)
    planner.tick()
    finish(planner)
    assert link.published[0]['goal_selection']['snapshot_id'] == 'fresh-run'


def test_stop_received_with_new_episode_still_wins():
    link, client = Link(), Client()
    planner = MathPlanner(link, client)
    link.command_received('stop')
    planner.tick()
    assert client.calls == link.published == []


def test_replaced_offer_invalidates_answer_and_keeps_only_one_worker():
    link, client = Link(), Client({'goal_id': 'A', 'reason': 'Old.'},
                                 {'goal_id': 'A', 'reason': 'Fresh.'}, block=True)
    planner = MathPlanner(link, client)
    planner.tick()
    assert client.entered.wait(3)
    old_request = planner._request
    link.state['goal_offer'] = offer('offer-2')
    planner.tick()
    assert len(client.calls) == 1
    client.release.set()
    assert old_request.done.wait(3)
    planner.tick()
    finish(planner)
    assert len(link.published) == 1
    assert link.published[0]['goal_selection']['snapshot_id'] == 'offer-2'


def test_invalid_answers_repair_once_then_use_labelled_budget_fallback():
    client = Client({'goal_id': 'bad', 'reason': 'Try.'},
                    {'goal_id': 'A', 'reason': 'Try.', 'velocity': 100})
    link, _, _ = run_choice(client=client)
    assert len(client.calls) == 2
    assert client.calls[1]['request']['validation_errors']
    assert link.published[0]['source'] == 'fallback'
    assert link.published[0]['goal_selection']['goal_id'] == 'A'
    assert len(link.entries[-1]['errors']) == 2


@pytest.mark.parametrize('error', [LLMUnavailable('SECRET_API_KEY'), RuntimeError('SECRET_API_KEY')])
def test_provider_failures_are_fallback_and_do_not_expose_response_bodies(error):
    link, client, _ = run_choice(client=Client(error))
    assert len(client.calls) == 1
    assert link.published[0]['source'] == 'fallback'
    assert 'SECRET_API_KEY' not in repr(link.published) + repr(link.entries)


@pytest.mark.parametrize('case', ['home_only', 'complete', 'emergency'])
def test_mandatory_return_uses_budget_without_model_call(case):
    link, client = Link(), Client()
    batch = link.state['goal_offer']
    if case == 'home_only':
        batch['candidates'][0]['feasible'] = False
    elif case == 'complete':
        batch['observations']['collected'] = 3
    else:
        batch['candidates'][1]['emergency'] = True
    planner = MathPlanner(link, client)
    planner.tick()
    assert client.calls == []
    assert link.published[0]['source'] == 'budget'
    assert link.published[0]['subgoals'] == [{'type': 'return_to_base'}]


def test_backend_rejection_is_feedback_for_a_fresh_offer_not_repeated_publish():
    client = Client({'goal_id': 'A', 'reason': 'First.'}, {'goal_id': 'A', 'reason': 'Repaired.'})
    link, _, planner = run_choice(client=client)
    link.status = {'plan_id': planner.inflight, 'state': 'failed', 'plan_complete': True,
                   'reason': 'Terrain changed; select from refreshed costs.', 'data': {'accepted': False}}
    planner.tick()
    assert planner.inflight is None
    assert len(link.published) == 1
    link.state['goal_offer'] = offer('revised-offer')
    planner.tick()
    finish(planner)
    assert client.calls[1]['request']['offer']['feedback'] == {
        'previous_rejection': 'Terrain changed; select from refreshed costs.'}
    assert link.published[1]['goal_selection']['snapshot_id'] == 'revised-offer'


def test_no_feasible_offer_stays_idle_and_reports_once():
    link, client = Link(), Client()
    for candidate in link.state['goal_offer']['candidates']:
        candidate['feasible'] = False
    planner = MathPlanner(link, client)
    planner.tick()
    planner.tick()
    assert client.calls == link.published == []
    assert len(link.entries) == 1
    assert link.entries[0]['status'] == 'rejected'


@pytest.mark.parametrize('corruption', ['wrong_episode', 'missing_steps', 'bad_cost', 'invalid_candidate'])
def test_incomplete_public_contract_never_calls_model(corruption):
    link, client = Link(), Client()
    batch = link.state['goal_offer']
    if corruption == 'wrong_episode':
        batch['episode_id'] = 99
    elif corruption == 'missing_steps':
        del batch['candidates'][0]['subgoals']
    elif corruption == 'bad_cost':
        batch['candidates'][0]['required_battery'] = float('nan')
    else:
        batch['candidates'].append('invalid')
    planner = MathPlanner(link, client)
    planner.tick()
    assert client.calls == link.published == []


def test_finished_episode_discards_late_response():
    link, client = Link(), Client(block=True)
    planner = MathPlanner(link, client)
    planner.tick()
    assert client.entered.wait(3)
    request = planner._request
    link.state['finished'] = True
    planner.tick()
    client.release.set()
    assert request.done.wait(3)
    planner.tick()
    assert link.published == []


def test_budget_stub_cannot_be_misreported_as_llm():
    link, _, _ = run_choice(client=BudgetStub())
    assert link.published[0]['source'] == 'fallback'


def test_budget_policy_selects_cheapest_feasible_search_without_calling_client():
    link, client = Link(), Client(RuntimeError('The baseline must never call a model.'))
    cheaper = deepcopy(link.state['goal_offer']['candidates'][0])
    cheaper.update(goal_id='B', required_battery=15.0)
    cheaper['subgoals'][0]['x'] = 0.8
    link.state['goal_offer']['candidates'].append(cheaper)
    planner = MathPlanner(link, client, selection_policy='budget')
    planner.tick()
    assert client.calls == []
    assert planner._request is None
    assert link.published[0]['source'] == 'budget'
    assert link.published[0]['goal_selection']['goal_id'] == 'B'
    assert link.published[0]['subgoals'] == cheaper['subgoals']
    assert link.entries[-1]['decision_source'] == 'budget'
    assert link.entries[-1]['errors'] == []


def test_budget_policy_needs_no_client_and_waits_for_whole_plan():
    link = Link()
    planner = MathPlanner(link, selection_policy='budget')
    planner.tick()
    plan_id = planner.inflight
    link.state['goal_offer'] = None
    link.state['current'] = {'plan_id': plan_id, 'index': 0, 'state': 'done'}
    link.status = {'plan_id': plan_id, 'index': 0, 'state': 'done', 'plan_complete': False}
    planner.tick()
    assert planner.inflight == plan_id
    assert len(link.published) == 1
    link.status = {'plan_id': plan_id, 'index': 1, 'state': 'done', 'plan_complete': True}
    link.state['goal_offer'] = offer('offer-2')
    planner.tick()
    assert len(link.published) == 2
    assert link.published[-1]['source'] == 'budget'
    assert link.published[-1]['plan_id'] != plan_id


def test_budget_policy_obeys_stop_and_does_not_repeat_consumed_offer():
    link = Link()
    planner = MathPlanner(link, selection_policy='budget')
    planner.tick()
    planner.tick()
    link.clock += 11
    planner.tick()
    assert len(link.published) == 1
    link.state['goal_offer'] = offer('offer-2')
    link.command_received('stop')
    planner.tick()
    assert len(link.published) == 1


@pytest.mark.parametrize('policy', ['stub', '', None])
def test_unknown_selection_policy_is_rejected(policy):
    with pytest.raises(ValueError, match='selection_policy must be llm or budget'):
        MathPlanner(Link(), Client(), selection_policy=policy)


def test_llm_policy_requires_an_explicit_client():
    with pytest.raises(ValueError, match='requires a model client'):
        MathPlanner(Link())


def test_metrics_include_only_allowlisted_model_counters():
    client = Client()
    client.cfg = SimpleNamespace(model='fake-model', api_key='SECRET')
    client.stats = lambda: {'calls_made': 4, 'cache_hits': 1, 'failed': 2,
                            'blocked_by_budget': 3, 'response': 'SECRET', 'api_key': 'SECRET'}
    metrics = MathPlanner(Link(), client).metrics()
    assert metrics == {'selection_policy': 'llm', 'model': 'fake-model', 'client_stats': {
        'calls_made': 4, 'cache_hits': 1, 'failed': 2, 'blocked_by_budget': 3}}
    assert 'SECRET' not in json.dumps(metrics)


def test_budget_metrics_are_zero_even_if_an_unused_client_has_counters():
    client = Client()
    client.cfg = SimpleNamespace(model='unused-model')
    client.stats = lambda: pytest.fail('Budget metrics must not inspect an unused client.')
    metrics = MathPlanner(Link(), client, selection_policy='budget').metrics()
    assert metrics['model'] is None
    assert metrics['selection_policy'] == 'budget'
    assert set(metrics['client_stats'].values()) == {0}
