"""Regression tests for the high-level planner prompt."""

import pytest

from did_llm.prompts import build_planner_prompt
from did_llm.prompts import build_planner_system


def test_prompt_uses_planner_limit_not_executor_ceiling():
    prompt = build_planner_system(6)

    assert 'Обычно 4–6 подцелей' in prompt
    assert 'максимум 6' in prompt
    assert 'максимум 50' not in prompt


def test_prompt_assigns_gradient_following_to_local_search():
    prompt = build_planner_system(6)
    example = prompt.split('ФОРМАТ ОТВЕТА', 1)[1]

    assert 'одно скалярное измерение без направления' in prompt
    assert 'Градиент вычисляет исполнитель внутри search_around' in prompt
    assert example.count('{"type": "collect"}') == 2
    assert 'управляешьроботом' not in prompt


@pytest.mark.parametrize(
    ('scenario', 'samples', 'profile', 'events'),
    [
        ('easy', 3, '1 зона', 'событий во время прогона нет'),
        ('medium', 5, '3 зоны', 'событий во время прогона нет'),
        ('hard', 7, '4 исходные зоны', 'появление новой опасной зоны'),
    ],
)
def test_prompt_explains_current_scenario_without_hidden_layout(
    scenario, samples, profile, events,
):
    prompt = build_planner_prompt(
        'Собрать образцы',
        {
            'scenario': scenario,
            'samples_total': samples,
            'collected': 1,
            'battery': 50.0,
        },
        None,
        '',
        budget={'search_budget': 15.0, 'cost_to_come_back': 3.0},
    )

    assert f'режим: {scenario}' in prompt
    assert f'всего {samples}, собрано 1, осталось {samples - 1}' in prompt
    assert profile in prompt
    assert events in prompt
    assert 'скрытые координаты неизвестны' in prompt


def test_seed_changes_layout_not_information_available_to_model():
    prompt = build_planner_prompt(
        'Собрать образцы',
        {'scenario': 'hard@7', 'samples_total': 7, 'collected': 0},
        None,
        '',
    )

    assert 'режим: hard@7' in prompt
    assert 'seed делает раскладку воспроизводимой' in prompt
    assert 'точные места и время этих событий тебе неизвестны' in prompt


def test_budget_is_described_as_spendable_work_not_battery_threshold():
    prompt = build_planner_prompt(
        'Собрать образцы',
        {'scenario': 'easy', 'samples_total': 3, 'collected': 0, 'battery': 60.0},
        None,
        '',
        budget={'search_budget': 24.0, 'cost_to_come_back': 0.0},
    )

    assert 'бюджет на новые поиски: 24.0' in prompt
    assert 'ниже 24.0 возвращаться' not in prompt


def test_unknown_ground_is_not_described_as_known_cheap_ground():
    prompt = build_planner_prompt(
        'Собрать образцы',
        {'scenario': 'medium', 'samples_total': 5, 'collected': 0},
        None,
        '',
        expensive=[],
    )

    assert 'не измерил ни одной дорогой зоны' in prompt
    assert 'все точки стоят 1.0' not in prompt


def test_plan_history_shows_recent_plans_and_outcomes():
    """The model sees what was already tried, so it does not repeat itself."""
    prompt = build_planner_prompt(
        'Собрать образцы',
        {'scenario': 'easy', 'samples_total': 3, 'collected': 0, 'battery': 60.0},
        None,
        '',
        plan_history=[
            {
                'plan_id': 'llm-001',
                'subgoals': [
                    {'type': 'goto', 'x': 0.5, 'y': 0.5},
                    {'type': 'search_around', 'x': 0.5, 'y': 0.5, 'radius': 0.8},
                    {'type': 'collect'},
                ],
                'source': 'llm',
                'outcome': 'failed',
                'reason': 'no path to goal',
            },
            {
                'plan_id': 'llm-002',
                'subgoals': [
                    {'type': 'search_around', 'x': -0.5, 'y': -0.5, 'radius': 0.8},
                    {'type': 'collect'},
                ],
                'source': 'llm',
                'outcome': 'done',
                'reason': '',
            },
        ],
    )

    assert 'ИСТОРИЯ ПЛАНОВ' in prompt
    assert 'llm-001' in prompt
    assert 'llm-002' in prompt
    assert 'no path to goal' in prompt
    assert 'Не повторяй планы' in prompt


def test_plan_history_is_empty_when_no_plans():
    """No history block when there are no plans yet."""
    prompt = build_planner_prompt(
        'Собрать образцы',
        {'scenario': 'easy', 'samples_total': 3, 'collected': 0, 'battery': 60.0},
        None,
        '',
        plan_history=[],
    )

    assert 'ИСТОРИЯ ПЛАНОВ' not in prompt


def test_expensive_ground_is_shown_in_prompt():
    """The model sees the cost map, so it can route around expensive zones."""
    prompt = build_planner_prompt(
        'Собрать образцы',
        {'scenario': 'medium', 'samples_total': 5, 'collected': 0, 'battery': 50.0},
        None,
        '',
        expensive=[
            {'x': 1.0, 'y': 1.0, 'reach': 0.5, 'cost': 2.5},
            {'x': -1.0, 'y': -1.0, 'reach': 0.3, 'cost': 3.0},
        ],
    )

    assert 'ДОРОГОЙ ГРУНТ' in prompt
    assert '(1.00; 1.00)' in prompt
    assert '×2.5' in prompt
    assert '(-1.00; -1.00)' in prompt
    assert '×3.0' in prompt
    assert 'Через эти точки не ездить' in prompt
