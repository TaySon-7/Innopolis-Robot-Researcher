import json

import pytest

from did_agent.plan import PlanError
from did_agent.plan import parse_plan


def plan_text(*subgoals, **extra):
    return json.dumps({'plan_id': 'p1', 'subgoals': list(subgoals), **extra})


def test_valid_plan_is_parsed():
    plan = parse_plan(plan_text(
        {'type': 'goto', 'x': -0.75, 'y': 0.25},
        {'type': 'search_around', 'x': 0, 'y': 0, 'radius': 0.8},
        {'type': 'collect'},
        {'type': 'return_to_base'},
    ))
    assert plan.plan_id == 'p1'
    assert [s.type for s in plan.subgoals] == [
        'goto', 'search_around', 'collect', 'return_to_base'
    ]
    assert plan.subgoals[1].radius == 0.8
    assert plan.subgoals[0].describe() == 'goto(-0.75, 0.25)'


def test_bare_list_is_accepted():
    plan = parse_plan(json.dumps([{'type': 'collect'}]))
    assert len(plan.subgoals) == 1


@pytest.mark.parametrize('text', [
    'not json at all',
    '42',
    '{"subgoals": "goto"}',
    '{}',
    plan_text({'type': 'fly', 'x': 1, 'y': 1}),
    plan_text({'type': 'goto', 'x': 1}),
    plan_text({'type': 'goto', 'x': 'one', 'y': 1}),
    plan_text({'type': 'goto', 'x': True, 'y': 1}),
    plan_text({'type': 'goto', 'x': 1e9, 'y': 1}),
    plan_text({'type': 'goto', 'x': float('nan'), 'y': 1}),
    plan_text({'type': 'search_around', 'x': 0, 'y': 0, 'radius': 0}),
    plan_text({'type': 'search_around', 'x': 0, 'y': 0, 'radius': 50}),
    plan_text('goto'),
    plan_text(*[{'type': 'collect'}] * 60),
])
def test_bad_plans_are_rejected_with_a_reason(text):
    with pytest.raises(PlanError) as error:
        parse_plan(text)
    assert str(error.value)


def test_none_input_is_a_plan_error():
    with pytest.raises(PlanError):
        parse_plan(None)
