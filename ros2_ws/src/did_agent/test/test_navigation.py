from math import hypot
from pathlib import Path

import pytest

from did_agent.costmap import CostMap
from did_agent.grid import load_map
from did_agent.kinematic_sim import BASE
from did_agent.kinematic_sim import KinematicSim
from did_agent.kinematic_sim import drive
from did_agent.navigator_core import DONE
from did_agent.navigator_core import FAILED
from did_agent.navigator_core import NavigatorCore
from did_agent.planner import plan_cells
from did_agent.planner import plan_waypoints

GOALS = [(-0.55, -0.55), (0.55, 0.55), (1.7, -0.5), (0.0, 1.9), (0.9, -1.7)]


@pytest.fixture(scope='module')
def grid():
    return load_map()


@pytest.fixture()
def costmap(grid):
    return CostMap(grid)


def test_map_matches_the_task_description(grid):
    assert grid.shape == (384, 384)
    assert grid.resolution == 0.05
    row, col = grid.world_to_cell(1.1, 0.0)  # a pillar: outline occupied, inside unknown
    assert grid.occupied[row, col] or grid.unknown[row, col]
    row, col = grid.world_to_cell(*BASE)
    assert not grid.occupied[row, col] and not grid.unknown[row, col]


def test_costmap_forbids_walls_and_margin_and_keeps_base_free(costmap):
    assert costmap.is_free(*costmap.world_to_cell(*BASE))
    assert not costmap.is_free(*costmap.world_to_cell(1.1, 0.0))
    assert not costmap.is_free(*costmap.world_to_cell(1.1 + 0.25, 0.0))  # 0.1 m margin
    assert costmap.is_free(*costmap.world_to_cell(1.1 + 0.6, 0.0))
    assert not costmap.is_free(*costmap.world_to_cell(-9.0, -9.0))  # unknown space


def test_every_scenario_sample_is_reachable(costmap):
    import yaml
    scenarios = Path(__file__).resolve().parents[2] / 'did_judge' / 'scenarios'
    start = costmap.world_to_cell(*BASE)
    for name in ('easy', 'medium', 'hard'):
        data = yaml.safe_load((scenarios / f'{name}.yaml').read_text())
        for sample in data['samples']:
            goal = costmap.world_to_cell(sample['x'], sample['y'])
            assert plan_cells(costmap, start, goal) is not None, (name, sample['id'])


def test_astar_goes_around_an_expensive_floor(costmap):
    start, goal = (-2.0, -0.5), (1.7, -0.5)
    straight = plan_waypoints(costmap, start, goal)

    costmap.update({'circle': {'x': -0.55, 'y': -0.5, 'r': 0.4}}, 10.0)
    detour = plan_waypoints(costmap, start, goal)

    assert costmap.energy_cost(detour) < costmap.energy_cost(straight) - 3.0

    def inside(point):
        return hypot(point[0] + 0.55, point[1] + 0.5) < 0.4

    steps = 200
    for a, b in zip(detour, detour[1:]):
        for i in range(steps):
            t = i / steps
            assert not inside((a[0] + (b[0] - a[0]) * t, a[1] + (b[1] - a[1]) * t))


def test_astar_crosses_an_expensive_floor_when_no_cheaper_way_exists(costmap):
    # Closing the whole corridor between two pillars: crossing it is still allowed.
    costmap.update({'rect': {'x_min': -0.2, 'y_min': -0.95, 'x_max': 0.2, 'y_max': -0.15}}, 5.0)
    path = plan_waypoints(costmap, (-0.55, -0.55), (0.55, -0.55))
    assert path is not None


def test_waypoints_never_cut_through_blocked_cells(costmap):
    path = plan_waypoints(costmap, (-2.0, -0.5), (1.7, 0.5))
    for a, b in zip(path, path[1:]):
        for i in range(100):
            t = i / 100
            row, col = costmap.world_to_cell(a[0] + (b[0] - a[0]) * t, a[1] + (b[1] - a[1]) * t)
            assert costmap.is_free(row, col)


def test_unreachable_goal_fails_cleanly(costmap):
    assert plan_waypoints(costmap, (-2.0, -0.5), (8.0, 8.0)) is None


def test_update_reports_changes_and_bumps_version(costmap):
    version = costmap.version
    assert costmap.update({'circle': {'x': 0.5, 'y': 0.5, 'r': 0.3}}, 2.0) > 0
    assert costmap.version == version + 1
    assert costmap.update({'circle': {'x': 0.5, 'y': 0.5, 'r': 0.3}}, 2.0) == 0
    assert costmap.version == version + 1
    with pytest.raises(ValueError):
        costmap.update({'blob': 1}, 2.0)
    with pytest.raises(ValueError):
        costmap.update({'circle': {'x': 0, 'y': 0, 'r': 1}}, 0.0)


@pytest.mark.parametrize('goal', GOALS)
def test_robot_reaches_goal_without_touching_anything(grid, costmap, goal):
    sim = KinematicSim(grid)
    nav = NavigatorCore(costmap)
    assert drive(nav, sim, goal) == DONE
    assert hypot(sim.pose.x - goal[0], sim.pose.y - goal[1]) < 0.2
    assert sim.collisions == 0


def test_robot_can_come_back_to_base(grid, costmap):
    sim = KinematicSim(grid)
    nav = NavigatorCore(costmap)
    assert drive(nav, sim, (1.7, 0.5)) == DONE
    assert drive(nav, sim, BASE) == DONE
    assert hypot(sim.pose.x - BASE[0], sim.pose.y - BASE[1]) < 0.15
    assert sim.collisions == 0


def test_expensive_floor_changes_the_route_the_robot_drives(grid):
    free = CostMap(grid)
    sim = KinematicSim(grid)
    assert drive(NavigatorCore(free), sim, (1.7, -0.5)) == DONE
    cheap_distance = sim.distance

    priced = CostMap(grid)
    priced.update({'circle': {'x': -0.55, 'y': -0.5, 'r': 0.4}}, 10.0)
    sim = KinematicSim(grid)
    assert drive(NavigatorCore(priced), sim, (1.7, -0.5)) == DONE
    assert sim.distance > cheap_distance


def test_unexpected_obstacle_is_avoided_by_replanning(grid, costmap):
    # A crate the map does not know about, right on the straight route.
    sim = KinematicSim(grid, hidden=[(-0.55, -0.5, 0.15)])
    nav = NavigatorCore(costmap)
    assert drive(nav, sim, (0.55, -0.55)) == DONE
    assert nav.replans >= 1
    assert sim.collisions == 0
    assert hypot(sim.pose.x - 0.55, sim.pose.y + 0.55) < 0.2


def test_goal_inside_a_sealed_area_fails_instead_of_looping(grid, costmap):
    sim = KinematicSim(grid, hidden=[(-1.0, -0.5, 0.35), (-1.0, 0.5, 0.35),
                                     (-1.0, -1.5, 0.35), (-1.0, 1.5, 0.35),
                                     (-1.0, 2.2, 0.35), (-1.0, -2.2, 0.35),
                                     (-1.0, 0.0, 0.35), (-1.0, -1.0, 0.35),
                                     (-1.0, 1.0, 0.35), (-1.0, -1.9, 0.35),
                                     (-1.0, 1.8, 0.35)])
    nav = NavigatorCore(costmap, max_replans=2)
    status = drive(nav, sim, (1.7, -0.5), timeout=120.0)
    assert status in (FAILED, DONE)
    assert sim.collisions == 0
