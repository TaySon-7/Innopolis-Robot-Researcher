import json
import urllib.error
import urllib.request

import numpy as np
import pytest

from did_agent.costmap import CostMap
from did_agent.dashboard_core import DashboardData
from did_agent.dashboard_core import DashboardServer
from did_agent.dashboard_core import costmap_layers
from did_agent.dashboard_core import render_geometry
from did_agent.dashboard_core import terrain_runs
from did_agent.dashboard_core import truth_from_scenario
from did_agent.grid import load_map
from did_agent.gazebo_geometry import _planar_axes
from did_agent.plan import parse_plan
from did_judge.scenario import load_scenario
from did_judge.scenario import scenario_path


@pytest.fixture(scope='module')
def geometry():
    return render_geometry(load_map())


def inside(polygon, x, y):
    """Ray casting point-in-polygon test."""
    hit = False
    for (x1, y1), (x2, y2) in zip(polygon, polygon[1:] + polygon[:1]):
        if (y1 > y) != (y2 > y) and x < (x2 - x1) * (y - y1) / (y2 - y1) + x1:
            hit = not hit
    return hit


def test_the_arena_outline_is_a_clean_polygon_not_a_staircase(geometry):
    floor = geometry['floor']
    assert 6 <= len(floor) <= 16            # the raw grid outline has hundreds of corners
    assert inside(floor, -2.0, -0.5)        # the base
    assert inside(floor, 0.0, 0.5)
    assert not inside(floor, 5.0, 5.0)
    xs, ys = [p[0] for p in floor], [p[1] for p in floor]
    assert -3.1 < min(xs) < -2.6 and 2.3 < max(xs) < 2.9
    assert -2.8 < min(ys) < -2.3 and 2.3 < max(ys) < 2.8


def test_there_are_nine_round_pillars_inside_the_arena(geometry):
    pillars = geometry['pillars']
    assert len(pillars) == 9
    for pillar in pillars:
        assert 0.1 <= pillar['r'] <= 0.2
        assert inside(geometry['floor'], pillar['x'], pillar['y'])
    xs = sorted({round(p['x'], 1) for p in pillars})
    assert len(xs) == 3 and abs(xs[1] - xs[0] - 1.1) < 0.15   # a 3x3 grid, 1.1 m apart


def test_bounds_frame_the_whole_arena_with_a_margin(geometry):
    b = geometry['bounds']
    xs, ys = [p[0] for p in geometry['floor']], [p[1] for p in geometry['floor']]
    assert b['xmin'] < min(xs) and b['xmax'] > max(xs)
    assert b['ymin'] < min(ys) and b['ymax'] > max(ys)
    assert geometry['resolution'] == 0.05 and geometry['origin'] == [-10.0, -10.0]
    assert 0.04 <= geometry['wall'] <= 0.2


def test_geometry_is_plain_json(geometry):
    json.dumps(geometry)


def test_gazebo_mesh_axes_keep_xy_order_when_y_span_is_wider():
    vertices = [(-2.0, 0.0, 0.0), (2.0, 0.0, 0.0), (0.0, -3.0, 0.0),
                (0.0, 3.0, 0.0)]
    assert _planar_axes(vertices) == (0, 1)


def test_terrain_runs_encode_only_priced_cells():
    costmap = CostMap()
    assert terrain_runs(costmap.terrain) == []
    costmap.update({'rect': {'x_min': 0.0, 'y_min': 0.0, 'x_max': 0.3, 'y_max': 0.1}}, 2.5)
    runs = terrain_runs(costmap.terrain)
    assert runs and all(v == 2.5 for _, _, _, v in runs)
    cells = sum(b - a + 1 for _, a, b, _ in runs)
    assert cells == int(np.count_nonzero(costmap.terrain != 1.0))


def test_costmap_display_has_separate_knowledge_planner_and_blocked_layers():
    costmap = CostMap()
    row, col = costmap.world_to_cell(-2.0, -0.5)
    costmap.last_seen[row, col:col + 3] = 12.0
    costmap.terrain[row, col] = 0.75
    layers = costmap_layers(costmap)
    assert layers['knowledge']
    assert any(run[3] == 0.75 for run in layers['knowledge'])
    assert layers['wall_cost'] and layers['total'] and layers['blocked']
    assert set(layers) == {
        'version', 'knowledge', 'terrain', 'wall_cost', 'total', 'blocked',
    }
    json.dumps(layers)


def test_data_store_keeps_a_thin_trail_and_marks_collections():
    data = DashboardData()
    for i in range(100):
        data.on_pose(-2.0 + i * 0.001, -0.5, 0.0)  # 10 cm in total
    assert len(data.snapshot()['trail']) <= 4
    data.on_event({'event': 'sample_collected', 't': 5.0})
    assert data.snapshot()['collected_at'] == [(-1.901, -0.5)]


def test_a_plan_explanation_becomes_a_journal_decision():
    data = DashboardData()
    data.on_plan(json.dumps({
        'plan_id': 'p3', 'explanation': 'Зона B дорогая, объезжаю',
        'subgoals': [{'type': 'collect'}],
    }))
    entry = data.snapshot()['journal'][0]
    assert entry['kind'] == 'decision' and 'объезжаю' in entry['text']
    data.on_plan('not json')  # must not raise


def test_truth_overlay_describes_the_scenario():
    truth = truth_from_scenario(load_scenario(scenario_path('hard')))
    assert len(truth['samples']) == 7 and len(truth['soil_zones']) == 4
    assert truth['base'] == {'x': -2.0, 'y': -0.5}


def test_truth_overlay_applies_silent_environment_events_at_their_time():
    scenario = load_scenario(scenario_path('hard'))
    before = truth_from_scenario(scenario, 89.9)
    changed = truth_from_scenario(scenario, 90.0)
    hazard = truth_from_scenario(scenario, 150.0)
    assert before['soil_zones'][0]['cost_multiplier'] == 2.0
    assert changed['soil_zones'][0]['cost_multiplier'] == 4.5
    assert changed['hazard_zones'] == []
    assert hazard['hazard_zones'][0]['id'] == 'h1'


@pytest.fixture()
def server(geometry):
    sent = {'plans': [], 'commands': []}
    srv = DashboardServer(
        DashboardData(), geometry,
        sent['plans'].append, sent['commands'].append, port=0, host='127.0.0.1',
    )
    srv.start()
    yield srv, sent
    srv.stop()


def call(srv, path, body=None):
    url = f'http://127.0.0.1:{srv.port}{path}'
    request = urllib.request.Request(
        url, data=None if body is None else json.dumps(body).encode(),
        headers={'Content-Type': 'application/json'},
    )
    try:
        with urllib.request.urlopen(request, timeout=5) as response:
            return response.status, response.read(), response.headers['Content-Type']
    except urllib.error.HTTPError as error:
        return error.code, error.read(), error.headers['Content-Type']


def test_server_serves_page_map_and_state(server):
    srv, _ = server
    status, body, kind = call(srv, '/')
    assert status == 200 and kind.startswith('text/html') and b'Robot Researcher' in body
    geometry = json.loads(call(srv, '/api/geometry')[1])
    assert len(geometry['pillars']) == 9 and geometry['floor']
    state = json.loads(call(srv, '/api/state')[1])
    assert {'pose', 'state', 'status', 'score', 'trail', 'events', 'journal', 'costmap'} <= set(state)
    assert call(srv, '/nope')[0] == 404


def test_click_sends_a_goto_plan(server):
    srv, sent = server
    status, body, _ = call(srv, '/api/goto', {'x': 0.5, 'y': -0.25})
    assert status == 200 and json.loads(body)['ok']
    plan = parse_plan(sent['plans'][-1])
    assert [s.type for s in plan.subgoals] == ['goto']
    assert (plan.subgoals[0].x, plan.subgoals[0].y) == (0.5, -0.25)


def test_shift_click_searches_and_collects(server):
    srv, sent = server
    call(srv, '/api/goto', {'x': 1.0, 'y': 1.0, 'mode': 'search'})
    plan = parse_plan(sent['plans'][-1])
    assert [s.type for s in plan.subgoals] == ['search_around', 'collect']


@pytest.mark.parametrize('body', [
    {'x': 'left', 'y': 0},
    {'x': 1},
    {'x': 1e9, 'y': 0},
    {'x': True, 'y': 0},
])
def test_bad_clicks_are_rejected_and_nothing_is_sent(server, body):
    srv, sent = server
    status, payload, _ = call(srv, '/api/goto', body)
    assert status == 400 and not json.loads(payload)['ok']
    assert json.loads(payload)['error']
    assert sent['plans'] == []


def test_raw_plans_are_validated_before_publishing(server):
    srv, sent = server
    assert call(srv, '/api/plan', {'subgoals': [{'type': 'collect'}]})[0] == 200
    assert call(srv, '/api/plan', {'subgoals': [{'type': 'dance'}]})[0] == 400
    assert len(sent['plans']) == 1


def test_commands_are_whitelisted(server):
    srv, sent = server
    assert call(srv, '/api/command', {'cmd': 'auto'})[0] == 200
    assert call(srv, '/api/command', {'cmd': 'stop'})[0] == 200
    assert call(srv, '/api/command', {'cmd': 'rm -rf'})[0] == 400
    assert sent['commands'] == ['auto', 'stop']


def test_garbage_bodies_do_not_crash_the_server(server):
    srv, _ = server
    url = f'http://127.0.0.1:{srv.port}/api/plan'
    request = urllib.request.Request(url, data=b'{not json')
    with pytest.raises(urllib.error.HTTPError) as error:
        urllib.request.urlopen(request, timeout=5)
    assert error.value.code == 400
    big = urllib.request.Request(url, data=b'x' * 30000)
    with pytest.raises(urllib.error.HTTPError) as error:
        urllib.request.urlopen(big, timeout=5)
    assert error.value.code == 413
    assert call(srv, '/api/state')[0] == 200  # still alive
