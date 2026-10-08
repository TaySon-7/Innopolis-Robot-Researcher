import math

import numpy as np
import pytest

from did_agent.adaptation import Adaptation
from did_agent.autonomous import AutonomousAgent
from did_agent.costmap import CostMap
from did_agent.learner import CostLearner
from did_agent.sim_robot import SimRobot
from did_agent.skills import Skills
from did_judge.scenario import load_scenario
from did_judge.scenario import scenario_path


def segment(cm, x0, x1, y, price):
    n = max(1, int(abs(x1 - x0) / 0.05))
    cells = []
    for i in range(n + 1):
        cell = list(cm.world_to_cell(x0 + (x1 - x0) * i / n, y))
        if cell not in cells:
            cells.append(cell)
    return {'distance': abs(x1 - x0), 'per_meter': price, 'cells': cells}


def drive_line(learner, cm, y, truth, x_from=-0.5, x_to=1.5, step=0.12):
    xs = np.arange(x_from, x_to, step)
    for a, b in zip(xs, xs[1:]):
        learner.observe(segment(cm, a, b, y, truth((a + b) / 2, y)))


def zone(cx, cy, r, price):
    return lambda x, y: price if math.hypot(x - cx, y - cy) <= r else 1.0


def terrain_at(cm, x, y):
    return float(cm.terrain[cm.world_to_cell(x, y)])


def test_a_priced_floor_is_learned_and_ordinary_floor_stays_ordinary():
    cm = CostMap()
    learner = CostLearner(cm, prior=0.05)
    truth = zone(0.5, 0.5, 0.4, 2.5)
    for _ in range(3):
        drive_line(learner, cm, 0.5, truth)
    assert 2.0 < terrain_at(cm, 0.5, 0.5) < 2.9
    assert terrain_at(cm, 1.4, 0.5) == pytest.approx(1.0, abs=0.1)
    assert terrain_at(cm, 0.5, 1.2) == 1.0       # never visited
    assert terrain_at(cm, -1.5, -1.5) == 1.0


def test_a_changed_floor_is_noticed_and_old_evidence_forgotten():
    cm = CostMap()
    learner = CostLearner(cm, prior=0.05)
    for _ in range(3):
        drive_line(learner, cm, 0.5, zone(0.5, 0.5, 0.4, 2.0))
    before = terrain_at(cm, 0.5, 0.5)
    for _ in range(3):
        drive_line(learner, cm, 0.5, zone(0.5, 0.5, 0.4, 4.5))
    after = terrain_at(cm, 0.5, 0.5)
    assert after > before + 1.2 and after > 3.4


def test_a_floor_that_became_cheap_again_is_noticed_too():
    cm = CostMap()
    learner = CostLearner(cm, prior=0.05)
    for _ in range(3):
        drive_line(learner, cm, 0.5, zone(0.5, 0.5, 0.4, 3.0))
    for _ in range(3):
        drive_line(learner, cm, 0.5, lambda x, y: 1.0)
    assert terrain_at(cm, 0.5, 0.5) < 1.5


@pytest.mark.parametrize('price', [40.0, 0.05, -2.0, float('nan')])
def test_outlier_segments_are_ignored(price):
    cm = CostMap()
    learner = CostLearner(cm)
    assert learner.observe(segment(cm, 0.0, 0.12, 0.5, price)) is False
    assert float(cm.terrain.max()) == 1.0


def test_set_terrain_bumps_the_version_only_for_material_changes():
    cm = CostMap()
    version = cm.version
    block = np.full((3, 3), 1.1, dtype=np.float32)
    assert cm.set_terrain(200, 200, block, threshold=0.2) is False
    assert cm.version == version and cm.terrain[200, 200] == pytest.approx(1.1)
    assert cm.set_terrain(200, 200, np.full((3, 3), 2.0, dtype=np.float32)) is True
    assert cm.version == version + 1


def test_external_estimates_become_evidence_the_learner_keeps():
    cm = CostMap()
    adaptation = Adaptation(cm)
    region = {'circle': {'x': 0.5, 'y': 0.5, 'r': 0.3}}
    adaptation.on_external_update(region, 3.0)
    assert terrain_at(cm, 0.5, 0.5) > 2.0
    # ordinary-floor measurements nearby do not erase it at once
    drive_line(adaptation.learner, cm, 0.5, lambda x, y: 1.0, x_from=1.0, x_to=1.5)
    assert terrain_at(cm, 0.5, 0.5) > 2.0


def test_a_hazard_hit_blocks_the_area_and_is_written_to_the_journal():
    cm = CostMap()
    journal = []
    adaptation = Adaptation(cm, lambda *entry: journal.append(entry))
    adaptation.on_battery(1.0, 50.0)
    adaptation.on_pose(1.0, 0.0, 0.5)
    adaptation.on_event(1.0, {'event': 'hazard_hit'})
    assert not cm.is_free(*cm.world_to_cell(0.0, 0.5))
    assert terrain_at(cm, 0.0, 0.5) > 5.0
    assert journal[0][0] == 'hypothesis' and 'Опасная зона' in journal[0][1]


def test_segments_just_after_a_penalty_are_not_learned_as_floor_price():
    cm = CostMap()
    adaptation = Adaptation(cm)
    adaptation.on_battery(0.0, 50.0)
    adaptation.on_pose(0.0, 0.0, 0.5)
    adaptation.on_event(0.5, {'event': 'collision'})
    battery, t = 50.0, 0.5
    for i in range(1, 12):
        t += 0.1
        battery -= 0.02 * 6.0           # looks like a very dear floor
        adaptation.on_battery(t, battery)
        adaptation.on_pose(t, i * 0.02, 0.5)
    assert float(cm.terrain.max()) == 1.0


def test_learning_can_be_switched_off():
    cm = CostMap()
    adaptation = Adaptation(cm, learn=False)
    adaptation.on_battery(0.0, 50.0)
    adaptation.on_pose(0.0, 0.0, 0.5)
    for i in range(1, 30):
        adaptation.on_battery(i * 0.1, 50.0 - i * 0.04)
        adaptation.on_pose(i * 0.1, i * 0.02, 0.5)
    assert float(cm.terrain.max()) == 1.0


def test_sensor_fault_is_journaled_and_cleared():
    import random
    rng = random.Random(3)
    journal = []
    adaptation = Adaptation(CostMap(), lambda *entry: journal.append(entry))
    t = 0.0
    for i in range(60):
        t += 0.1
        adaptation.on_sensor(t, 0.3 + 0.002 * i + rng.gauss(0, 0.01))
    assert journal == []
    for _ in range(40):
        t += 0.1
        adaptation.on_sensor(t, 0.5 + rng.gauss(0, 0.15))
    assert any('Шум датчика' in entry[1] and entry[3] == 'open' for entry in journal)
    for _ in range(400):
        t += 0.1
        adaptation.on_sensor(t, 0.3 + rng.gauss(0, 0.01))
    assert any(entry[2] == '' and entry[3] == 'confirmed' for entry in journal)


# --- whole episodes with learning --------------------------------------------------------


def scenario(name):
    return load_scenario(scenario_path(name))


def test_driving_through_a_dear_zone_teaches_the_agent_about_it():
    robot = SimRobot(scenario('hard'))
    skills = Skills(robot)
    for _ in range(2):
        assert skills.goto(1.7, -0.45).ok
        assert skills.goto(1.7, -1.35).ok
    z3 = terrain_at(robot.costmap, 1.7, -0.9)   # truth: x4 in a r=0.4 circle
    assert z3 > 2.0
    assert terrain_at(robot.costmap, -1.5, 1.5) == 1.0
    assert any('Дорогой участок' in e['title'] for e in robot.journal_log)


def test_the_return_estimate_follows_what_was_learned():
    robot = SimRobot(scenario('hard'))
    skills = Skills(robot)
    skills.goto(1.7, -1.35)
    skills.goto(1.7, -0.45)
    skills.goto(1.7, -1.35)
    fresh = SimRobot(scenario('hard'))
    fresh.sim.pose.x, fresh.sim.pose.y = robot.sim.pose.x, robot.sim.pose.y
    assert skills.return_cost_estimate() >= Skills(fresh).return_cost_estimate()


def test_hard_episode_with_learning_is_complete_and_clean():
    robot = SimRobot(scenario('hard'))
    summary = AutonomousAgent(robot).run()
    assert summary['collected'] >= 6
    assert summary['returned_to_base']
    assert robot.judge.collisions == 0
    assert robot.judge.hazard_hits <= 1
    assert robot.adaptation.hypotheses >= 2
    assert robot.battery() > 5.0


def test_anomalies_make_the_agent_go_home_earlier():
    robot = SimRobot(scenario('hard'))
    skills = Skills(robot)
    skills.goto(1.7, -0.45)
    cost = skills.return_cost_estimate()
    robot.anomaly = lambda: {}  # calm times
    # a battery level that is enough in calm times, but not with a widened margin
    robot.judge.battery = cost * skills.return_factor + skills.reserve + 0.5
    assert skills.battery_allows_more()
    robot.anomaly = lambda: {'battery_deviation': True}
    assert not skills.battery_allows_more()


# Prices stay inside the range the generator can produce (after a silent soil change
# the dearest floor is x5 x1.3). Far beyond it, an unvisited changed floor can eat the reserve.
@pytest.mark.parametrize(('battery', 'multiplier'), [(45, 1.3), (35, 1.3), (28, 1.0)])
def test_a_thin_battery_and_dear_floors_still_end_at_the_base(battery, multiplier):
    sc = scenario('hard')
    sc.initial_battery = float(battery)
    for soil in sc.soil_zones:
        soil.cost_multiplier *= multiplier
    robot = SimRobot(sc)
    summary = AutonomousAgent(robot).run()
    assert summary['returned_to_base']
    assert robot.battery() > 0.5
    assert robot.judge.collisions == 0
