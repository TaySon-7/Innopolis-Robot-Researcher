from copy import deepcopy

import pytest
import yaml

from did_agent.scenario_generator import SPECS
from did_agent.scenario_generator import ScenarioError
from did_agent.scenario_generator import World
from did_agent.scenario_generator import generate
from did_agent.scenario_generator import load_named
from did_agent.scenario_generator import main
from did_agent.scenario_generator import to_yaml
from did_agent.scenario_generator import tour_cost
from did_agent.scenario_generator import validate
from did_judge.scenario import Zone
from did_judge.scenario import load_scenario
from did_judge.scenario import scenario_from_dict
from did_judge.scenario import scenario_path
from did_judge.scenario import scenario_to_dict


@pytest.fixture(scope='module')
def world():
    return World()


@pytest.mark.parametrize('name', ['easy', 'medium', 'hard'])
def test_the_shipped_scenarios_pass_the_validator(world, name):
    assert validate(load_scenario(scenario_path(name)), world) == []


@pytest.mark.parametrize('difficulty', ['easy', 'medium', 'hard'])
def test_generated_scenarios_follow_the_task_and_are_valid(world, difficulty):
    samples, soils, silent = SPECS[difficulty]
    for seed in range(1, 6):
        scenario = generate(difficulty, seed, world)
        assert len(scenario.samples) == samples
        assert len(scenario.soil_zones) == soils
        assert bool(scenario.events) == silent
        assert validate(scenario, world) == []
        assert scenario.name == f'{difficulty}@{seed}'


def test_hard_has_all_three_silent_events_in_time_order(world):
    scenario = generate('hard', 4, world)
    assert sorted(e.type for e in scenario.events) == ['hazard_appear', 'sensor_fault', 'soil_change']
    times = [e.at for e in scenario.events]
    assert times == sorted(times)


def test_the_same_seed_gives_the_same_scenario_and_other_seeds_differ(world):
    first = to_yaml(generate('medium', 7, world))
    assert first == to_yaml(generate('medium', 7, world))
    assert first != to_yaml(generate('medium', 8, world)).replace('medium@8', 'medium@7')


def test_overrides_change_the_counts(world):
    scenario = generate('easy', 2, world, samples=5, soils=2)
    assert len(scenario.samples) == 5 and len(scenario.soil_zones) == 2


def test_unknown_difficulty_is_an_error(world):
    with pytest.raises(ScenarioError):
        generate('impossible', 1, world)


def test_generated_yaml_is_read_back_by_the_judge_loader(world, tmp_path):
    scenario = generate('hard', 3, world)
    path = tmp_path / 'hard3.yaml'
    path.write_text(to_yaml(scenario), encoding='utf-8')
    assert scenario_to_dict(load_scenario(path)) == scenario_to_dict(scenario)


def test_load_named_reads_bundled_and_generates_seeded(world):
    assert load_named('hard', world).name == 'hard'
    assert load_named('hard@5', world).name == 'hard@5'
    with pytest.raises(ScenarioError):
        load_named('hard@x', world)


# --- the validator must catch bad scenarios -----------------------------------------------


def base_data():
    return deepcopy(scenario_to_dict(load_scenario(scenario_path('medium'))))


def problems(world, data):
    return validate(scenario_from_dict(data), world)


def test_a_sample_inside_a_pillar_is_rejected(world):
    data = base_data()
    data['samples'][0].update(x=1.1, y=0.0)
    assert any('sample s1' in p for p in problems(world, data))


def test_a_sample_next_to_a_wall_is_rejected(world):
    data = base_data()
    data['samples'][0].update(x=-2.6, y=0.0)
    assert any('s1' in p and ('wall' in p or 'reachable' in p) for p in problems(world, data))


def test_a_sample_outside_the_arena_is_rejected(world):
    data = base_data()
    data['samples'][0].update(x=6.0, y=6.0)
    assert any('not reachable' in p for p in problems(world, data))


def test_two_samples_on_top_of_each_other_are_rejected(world):
    data = base_data()
    data['samples'][1].update(x=data['samples'][0]['x'] + 0.1, y=data['samples'][0]['y'])
    assert any('too close to each other' in p for p in problems(world, data))


def test_a_hazard_over_the_base_or_a_sample_is_rejected(world):
    data = base_data()
    data['hazard_zones'] = [{'id': 'h1', 'shape': 'circle', 'x': -2.0, 'y': -0.5,
                             'radius': 0.4, 'penalty': 5.0}]
    assert any('covers the base' in p for p in problems(world, data))
    sample = data['samples'][0]
    data['hazard_zones'] = [{'id': 'h1', 'shape': 'circle', 'x': sample['x'], 'y': sample['y'],
                             'radius': 0.3, 'penalty': 5.0}]
    assert any('covers sample s1' in p for p in problems(world, data))


def test_duplicate_ids_are_rejected(world):
    data = base_data()
    data['samples'][1]['id'] = data['samples'][0]['id']
    assert any('duplicate' in p for p in problems(world, data))


def test_a_tour_that_does_not_fit_the_battery_is_rejected(world):
    data = base_data()
    data['battery']['initial'] = 12.0
    assert any('battery' in p for p in problems(world, data))


def test_tour_cost_grows_with_the_price_of_the_floor(world):
    scenario = load_scenario(scenario_path('hard'))
    cheap = tour_cost(scenario, world)
    for zone in scenario.soil_zones:
        zone.cost_multiplier *= 3
    assert tour_cost(scenario, world) > cheap


def test_command_line_writes_a_valid_file_and_checks_it(tmp_path, capsys):
    out = tmp_path / 'made.yaml'
    main(['--difficulty', 'medium', '--seed', '9', '--out', str(out)])
    assert out.exists() and 'medium@9' in capsys.readouterr().out
    with pytest.raises(SystemExit) as ok:
        main(['--check', str(out)])
    assert ok.value.code == 0
    broken = yaml.safe_load(out.read_text())
    broken['samples'][0].update(x=6.0, y=6.0)
    bad = tmp_path / 'bad.yaml'
    bad.write_text(yaml.safe_dump(broken))
    with pytest.raises(SystemExit) as fail:
        main(['--check', str(bad)])
    assert fail.value.code == 1


def test_a_zone_with_an_empty_shape_cannot_even_be_loaded():
    with pytest.raises(ValueError):
        Zone  # noqa: B018 - keep the import used
        data = base_data()
        data['soil_zones'][0] = {'id': 'z1', 'shape': 'circle', 'x': 0, 'y': 0, 'radius': 0}
        scenario_from_dict(data)


@pytest.mark.parametrize('name', ['easy@11', 'medium@12', 'medium@13', 'hard@14', 'hard@15'])
def test_the_agent_copes_with_scenarios_it_was_never_tuned_on(name):
    from did_agent.bench import run_scenario
    summary = run_scenario(name)
    assert summary['returned_to_base'] and summary['finished']
    assert summary['collisions'] == 0
    assert summary['battery'] > 0.5
    expected = summary['samples_total']
    assert summary['collected'] >= (expected if name.startswith(('easy', 'medium')) else expected - 2)
