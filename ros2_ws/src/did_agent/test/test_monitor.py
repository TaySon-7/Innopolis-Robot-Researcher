from did_agent.monitor import Monitor


def drive(monitor, per_meter, metres, start_t=0.0, battery=60.0, speed=0.2):
    """Feed a straight drive along x at 10 Hz with a given battery use per metre."""
    segments = []
    x, t = 0.0, start_t
    monitor.on_battery(t, battery)
    monitor.on_pose(t, x, 0.0)
    while x < metres:
        x += speed * 0.1
        t += 0.1
        battery -= speed * 0.1 * per_meter
        monitor.on_battery(t, battery)
        segment = monitor.on_pose(t, x, 0.0)
        if segment:
            segments.append(segment)
    return segments, t


def test_segments_report_the_measured_battery_use_per_metre():
    monitor = Monitor()
    segments, _ = drive(monitor, per_meter=2.0, metres=1.0)
    assert len(segments) >= 3
    for segment in segments:
        assert abs(segment['per_meter'] - 2.0) < 0.2
        assert 0.2 < segment['distance'] < 0.32


def test_a_price_the_agent_does_not_expect_raises_the_flag_after_two_segments():
    monitor = Monitor(expected_cost=lambda x, y: 1.0)
    _, t = drive(monitor, per_meter=2.5, metres=0.4)
    assert monitor.anomaly(t)['battery_deviation'] is False  # one segment is not enough
    _, t = drive(monitor, per_meter=2.5, metres=1.0, start_t=t, battery=50.0)
    assert monitor.anomaly(t)['battery_deviation'] is True


def test_expected_price_means_no_flag():
    monitor = Monitor(expected_cost=lambda x, y: 2.5)
    _, t = drive(monitor, per_meter=2.5, metres=2.0)
    assert monitor.anomaly(t)['battery_deviation'] is False


def test_flag_expires():
    monitor = Monitor(anomaly_hold=5.0)
    _, t = drive(monitor, per_meter=3.0, metres=1.5)
    assert monitor.anomaly(t)['battery_deviation']
    assert not monitor.anomaly(t + 6.0)['battery_deviation']


def test_two_penalties_in_a_short_time_are_a_burst():
    monitor = Monitor()
    monitor.on_event(10.0, {'event': 'collision'})
    assert not monitor.anomaly(11.0)['penalties_burst']
    monitor.on_event(12.0, {'event': 'hazard_hit'})
    assert monitor.anomaly(13.0)['penalties_burst']
    assert not monitor.anomaly(60.0)['penalties_burst']
    monitor.on_event(61.0, {'event': 'sample_collected'})
    assert not monitor.anomaly(62.0)['penalties_burst']


def test_sensor_noise_jump_is_detected_but_a_smooth_signal_is_not():
    import random
    rng = random.Random(1)
    monitor = Monitor()
    t = 0.0
    for i in range(60):  # smooth approach with little noise
        t += 0.1
        monitor.on_sensor(t, 0.2 + 0.004 * i + rng.gauss(0, 0.01))
    assert not monitor.anomaly(t)['sensor_noise_up']
    assert monitor.noise_estimate < 0.03

    for _ in range(40):  # sensor fault
        t += 0.1
        monitor.on_sensor(t, 0.5 + rng.gauss(0, 0.15))
    assert monitor.anomaly(t)['sensor_noise_up']
    assert monitor.noise_estimate > 0.08
