from did_judge.judge_model import JudgeModel


def make_model():
    return JudgeModel(
        base_x=-2.0,
        base_y=-0.5,
        initial_battery=60.0,
        battery_cost_per_meter=2.0,
        collection_radius=0.30,
        sensor_range=1.50,
        sensor_noise_stddev=0.0,
        random_seed=1,
        sample_positions=[-1.5, -0.5, 0.0, 0.0],
    )


def test_odometry_uses_world_offset_and_drains_battery():
    model = make_model()
    model.update_odometry(0.0, 0.0)
    model.update_odometry(0.25, 0.0)

    assert model.world_x == -1.75
    assert model.world_y == -0.5
    assert model.distance_travelled == 0.25
    assert model.battery == 59.5


def test_sample_sensor_and_collection():
    model = make_model()
    model.update_odometry(0.25, 0.0)

    assert model.sample_sensor() > 0.8
    assert model.collect()
    assert model.collected_count == 1
    assert not model.collect()


def test_finish_requires_return_to_base():
    model = make_model()
    model.update_odometry(1.0, 0.0)
    assert not model.at_base()

    model.update_odometry(0.1, 0.0)
    assert model.at_base()

