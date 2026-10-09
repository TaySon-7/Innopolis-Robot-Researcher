"""check_plan must report every problem at once, not just the first.

The contract in the docstring — "Returns a list of messages rather than the
first one: telling the model about all the bad points at once costs one call
instead of one per point" — was violated by two loops that broke out of the
subgoal scan on the first message in the shared list. The repair round then
got an incomplete diagnosis, fixed exactly what it was told, and came back
rejected on the problem that had been skipped.
"""

from did_llm.agent_plan import Plan
from did_llm.agent_plan import Subgoal
from did_llm.agent_plan import check_plan

#: Ground the mission forbids the model to drive onto.
EXPENSIVE = [{'x': -2.0, 'y': -0.5, 'reach': 0.3, 'cost': 3.0}]


def make_plan(*subgoals: dict) -> Plan:
    return Plan(plan_id='t', subgoals=[Subgoal(**item) for item in subgoals])


def test_clean_plan_passes():
    plan = make_plan({'type': 'goto', 'x': 1.5, 'y': -0.5})

    assert check_plan(plan, expensive=EXPENSIVE) == []


def test_expensive_ground_is_reported_despite_an_earlier_problem():
    # Subgoal 0 is outside the arena, subgoal 1 sits on dear ground: the old
    # check saw the arena message in the shared list and stopped scanning
    # after the first subgoal, so the ground problem never reached the model.
    plan = make_plan(
        {'type': 'goto', 'x': 2.5, 'y': -0.5},   # beyond hi_x = 2.45
        {'type': 'goto', 'x': -2.0, 'y': -0.5},  # on the expensive patch
    )

    problems = check_plan(plan, expensive=EXPENSIVE)

    assert any('вне арены' in item for item in problems)
    assert any('дорогом грунте' in item for item in problems)


def test_every_subgoal_on_dear_ground_gets_its_own_message():
    plan = make_plan(
        {'type': 'goto', 'x': -2.0, 'y': -0.5},
        {'type': 'goto', 'x': -2.0, 'y': -0.75},  # 0.25 m from the patch centre
    )

    problems = check_plan(plan, expensive=EXPENSIVE)

    assert sum('дорогом грунте' in item for item in problems) == 2


def test_every_known_collision_site_is_reported():
    plan = make_plan(
        {'type': 'goto', 'x': -2.0, 'y': -0.5},
        {'type': 'goto', 'x': -2.0, 'y': -0.75},
    )

    problems = check_plan(plan, hits=[(-2.0, -0.5), (-2.0, -0.75)])

    assert sum('столкновение' in item for item in problems) == 2


def test_search_under_the_robot_is_still_allowed_at_a_collision_site():
    # A search where the robot stands is how it walks off an obstacle; the
    # veto must keep letting it happen even with a known hit underneath.
    plan = make_plan(
        {'type': 'search_around', 'x': -2.0, 'y': -0.5, 'radius': 0.4},
        {'type': 'collect'},
    )

    problems = check_plan(
        plan,
        hits=[(-2.0, -0.5)],
        pose=(-2.0, -0.5),
        signal_high=0.85,
    )

    assert problems == []
