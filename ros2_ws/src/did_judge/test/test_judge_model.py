from pathlib import Path

import pytest

from did_judge.judge_model import JudgeModel
from did_judge.scenario import load_scenario
from did_judge.scenario import scenario_from_dict

SCENARIOS = Path(__file__).resolve().parents[1] / 'scenarios'


def make_scenario(**overrides):
    data = {
        'name': 'test',
        'seed': 1,
        'base': {'x': -2.0, 'y': -0.5},
        'battery': {'initial': 60.0, 'cost_per_meter': 2.0},
        'collection_radius': 0.30,
        'sensor': {'range': 1.5, 'noise_stddev': 0.0},
        'samples': [{'id': 's1', 'x': -1.5, 'y': -0.5}, {'id': 's2', 'x': 0.0, 'y': 0.0}],
        'soil_zones': [],
        'hazard_zones': [],
        'events': [],
    }
    data.update(overrides)
    return scenario_from_dict(data)


def make_model(**overrides):
    return JudgeModel(make_scenario(**overrides))


def test_odometry_uses_world_offset_and_drains_battery():
    model = make_model()
    model.update_odometry(0.0, 0.0)
    model.update_odometry(0.25, 0.0)

    assert model.world_x == -1.75
    assert model.world_y == -0.5
    assert model.distance_travelled == 0.25
    assert model.battery == 59.5


def test_soil_zone_multiplies_battery_use():
    zone = {'id': 'z1', 'shape': 'circle', 'x': -1.5, 'y': -0.5, 'radius': 0.2,
            'cost_multiplier': 3.0}
    model = make_model(soil_zones=[zone])
    model.update_odometry(0.0, 0.0)
    model.update_odometry(0.5, 0.0)  # step midpoint (-1.75, -0.5) is outside
    plain = 60.0 - 0.5 * 2.0
    assert model.battery == pytest.approx(plain)

    model.update_odometry(0.5 + 0.2, 0.0)  # midpoint (-1.4, -0.5) is inside
    assert model.battery == pytest.approx(plain - 0.2 * 2.0 * 3.0)


def test_soil_change_event_is_silent_and_changes_cost():
    zone = {'id': 'z1', 'shape': 'rect', 'x_min': -2.1, 'y_min': -0.7,
            'x_max': 1.0, 'y_max': 0.0, 'cost_multiplier': 2.0}
    event = {'at': 10.0, 'type': 'soil_change', 'zone': 'z1', 'cost_multiplier': 5.0}
    model = make_model(soil_zones=[zone], events=[event])
    model.update_odometry(0.0, 0.0)
    model.update_odometry(0.1, 0.0)
    before = 60.0 - 0.1 * 2.0 * 2.0
    assert model.battery == pytest.approx(before)

    model.advance(10.0)
    model.update_odometry(0.2, 0.0)
    assert model.battery == pytest.approx(before - 0.1 * 2.0 * 5.0)
    assert model.environment_log[0]['type'] == 'soil_change'


def test_hazard_hit_once_per_entry_and_after_appearing():
    hazard = {'id': 'h1', 'shape': 'circle', 'x': -1.5, 'y': -0.5, 'radius': 0.2,
              'penalty': 5.0}
    model = make_model(hazard_zones=[hazard])
    model.update_odometry(0.0, 0.0)
    assert model.update_odometry(0.45, 0.0) == [{'event': 'hazard_hit', 'zone': 'h1'}]
    assert model.update_odometry(0.50, 0.0) == []  # still inside
    assert model.hazard_hits == 1
    assert model.battery < 60.0 - 5.0

    model.update_odometry(1.5, 0.0)  # leave
    assert model.update_odometry(0.45, 0.0) != []  # enter again
    assert model.hazard_hits == 2


def test_hazard_appear_event_hits_a_robot_standing_inside():
    zone = {'id': 'h2', 'shape': 'circle', 'x': -2.0, 'y': -0.5, 'radius': 0.3,
            'penalty': 1.0}
    model = make_model(events=[{'at': 5.0, 'type': 'hazard_appear', 'zone': zone}])
    model.update_odometry(0.0, 0.0)
    assert model.hazard_hits == 0
    model.advance(5.0)
    assert model.update_odometry(0.0, 0.0)[0]['zone'] == 'h2'


def test_sensor_fault_raises_noise_for_its_duration():
    model = make_model(
        events=[{'at': 5.0, 'type': 'sensor_fault', 'noise_stddev': 0.3,
                 'duration': 10.0}],
    )
    assert model.sensor_noise_stddev == 0.0
    model.advance(5.0)
    assert model.sensor_noise_stddev == 0.3
    model.advance(15.0)
    assert model.sensor_noise_stddev == 0.0


def test_collision_is_rate_limited():
    model = make_model()
    model.advance(1.0)
    assert model.report_scan(0.3) is None
    assert model.report_scan(0.0)['event'] == 'collision'
    model.advance(1.5)
    assert model.report_scan(0.0) is None
    model.advance(3.5)
    assert model.report_scan(0.0) is not None
    assert model.collisions == 2


def test_sample_sensor_and_collection():
    model = make_model()
    model.update_odometry(0.25, 0.0)

    assert model.sample_sensor() > 0.8
    assert model.collect()[0]
    assert model.collected_count == 1
    ok, event = model.collect()
    assert not ok and event['event'] == 'false_collect'
    assert model.false_collects == 1


def test_finish_requires_return_to_base():
    model = make_model()
    model.update_odometry(1.0, 0.0)
    assert not model.finish()

    model.update_odometry(0.1, 0.0)
    assert model.finish()
    assert model.finished


def test_score_combines_rewards_and_penalties():
    model = make_model()
    model.update_odometry(0.25, 0.0)
    model.collect()
    model.collect()
    assert model.score == pytest.approx(10.0 - 1.0)


def test_same_seed_gives_same_noise():
    first = make_model(sensor={'range': 1.5, 'noise_stddev': 0.05})
    second = make_model(sensor={'range': 1.5, 'noise_stddev': 0.05})
    assert [first.sample_sensor() for _ in range(5)] == [
        second.sample_sensor() for _ in range(5)
    ]


@pytest.mark.parametrize(
    ('name', 'samples', 'soils', 'has_events'),
    [('easy', 3, 1, False), ('medium', 5, 3, False), ('hard', 7, 4, True)],
)
def test_shipped_scenarios_match_the_task(name, samples, soils, has_events):
    scenario = load_scenario(SCENARIOS / f'{name}.yaml')
    assert len(scenario.samples) == samples
    assert len(scenario.soil_zones) == soils
    assert bool(scenario.events) == has_events


def test_bad_scenarios_are_rejected():
    with pytest.raises(ValueError):
        make_scenario(soil_zones=[{'id': 'z', 'shape': 'blob'}])
    with pytest.raises(ValueError):
        make_scenario(events=[{'at': 1.0, 'type': 'soil_change', 'zone': 'nope',
                               'cost_multiplier': 2.0}])
    with pytest.raises(ValueError):
        make_scenario(events=[{'at': 1.0, 'type': 'earthquake'}])


def test_clearance_depends_on_the_side_the_obstacle_is_on():
    from math import pi
    from did_judge.judge_model import scan_clearance
    # A wall 0.13 m from the lidar: touching ahead, but 2 cm of air at the side.
    assert scan_clearance([0.0], [0.13]) < 0.015
    assert scan_clearance([pi / 2], [0.13]) > 0.015
    assert scan_clearance([0.0], [float('inf')]) is None
    assert scan_clearance([0.0], [0.05], range_min=0.12) is None  # below lidar range
