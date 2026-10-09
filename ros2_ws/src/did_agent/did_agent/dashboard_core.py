"""Dashboard backend without ROS: data store, map image and HTTP server."""

from __future__ import annotations

from collections import deque
from dataclasses import asdict
from http.server import BaseHTTPRequestHandler
from http.server import ThreadingHTTPServer
from math import hypot
from math import isfinite
from pathlib import Path
import json
import re
import threading
from typing import Any
from typing import Callable

import numpy as np

from did_agent.grid import GridMap
from did_agent.plan import PlanError
from did_agent.plan import parse_plan

WEB_DIR_SOURCE = Path(__file__).resolve().parent.parent / 'web'
COMMANDS = ('auto', 'stop', 'llm')
NAVIGATION_BACKENDS = ('custom', 'nav2')
SCENARIOS = ('easy', 'medium', 'hard')
MAX_SCENARIO_SEED = 2_147_483_647
SCENARIO_NAME = re.compile(r'^(easy|medium|hard)@(0|[1-9][0-9]{0,9})$')
MAX_BODY = 20_000


def valid_scenario_name(value: Any) -> bool:
    """Return whether ``value`` is a built-in name or a bounded seeded name."""
    if not isinstance(value, str):
        return False
    if value in SCENARIOS:
        return True
    match = SCENARIO_NAME.fullmatch(value)
    return match is not None and int(match.group(2)) <= MAX_SCENARIO_SEED


def web_dir() -> Path:
    """Return the folder with the page: installed share directory, else source tree."""
    try:
        from ament_index_python.packages import get_package_share_directory
        installed = Path(get_package_share_directory('did_agent')) / 'web'
        if (installed / 'index.html').exists():
            return installed
    except Exception:  # noqa: BLE001 - plain Python run without ROS installed
        pass
    return WEB_DIR_SOURCE


# --- map geometry -------------------------------------------------------------------


def _outside(grid: GridMap) -> np.ndarray:
    """Cells of unknown space connected to the map border (not pillar interiors)."""
    unknown = grid.unknown
    rows, cols = unknown.shape
    outside = np.zeros_like(unknown)
    queue = deque()
    for r in range(rows):
        for c in (0, cols - 1):
            if unknown[r, c] and not outside[r, c]:
                outside[r, c] = True
                queue.append((r, c))
    for c in range(cols):
        for r in (0, rows - 1):
            if unknown[r, c] and not outside[r, c]:
                outside[r, c] = True
                queue.append((r, c))
    while queue:
        r, c = queue.popleft()
        for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            a, b = r + dr, c + dc
            if 0 <= a < rows and 0 <= b < cols and unknown[a, b] and not outside[a, b]:
                outside[a, b] = True
                queue.append((a, b))
    return outside


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    """8-connected components of a boolean mask."""
    seen = np.zeros_like(mask)
    rows, cols = mask.shape
    found = []
    for r, c in zip(*np.where(mask)):
        if seen[r, c]:
            continue
        queue, cells = deque([(int(r), int(c))]), []
        seen[r, c] = True
        while queue:
            a, b = queue.popleft()
            cells.append((a, b))
            for da in (-1, 0, 1):
                for db in (-1, 0, 1):
                    x, y = a + da, b + db
                    if 0 <= x < rows and 0 <= y < cols and mask[x, y] and not seen[x, y]:
                        seen[x, y] = True
                        queue.append((x, y))
        found.append(cells)
    return found


def convex_hull(points: list[tuple[float, float]]) -> list[tuple[float, float]]:
    """Andrew's monotone chain; returns the hull counter-clockwise."""
    pts = sorted(set(points))
    if len(pts) < 3:
        return pts

    def cross(o, a, b):
        return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])

    lower: list[tuple[float, float]] = []
    for p in pts:
        while len(lower) >= 2 and cross(lower[-2], lower[-1], p) <= 0:
            lower.pop()
        lower.append(p)
    upper: list[tuple[float, float]] = []
    for p in reversed(pts):
        while len(upper) >= 2 and cross(upper[-2], upper[-1], p) <= 0:
            upper.pop()
        upper.append(p)
    return lower[:-1] + upper[:-1]


def simplify_polygon(points: list[tuple[float, float]], tolerance: float) -> list[tuple[float, float]]:
    """Ramer-Douglas-Peucker for a closed polygon (split at the two farthest points)."""
    if len(points) <= 4:
        return points

    def line_distance(p, a, b):
        dx, dy = b[0] - a[0], b[1] - a[1]
        length = hypot(dx, dy)
        if length == 0:
            return hypot(p[0] - a[0], p[1] - a[1])
        return abs(dx * (a[1] - p[1]) - dy * (a[0] - p[0])) / length

    def rdp(chain):
        if len(chain) < 3:
            return chain
        index, far = 0, 0.0
        for i in range(1, len(chain) - 1):
            d = line_distance(chain[i], chain[0], chain[-1])
            if d > far:
                index, far = i, d
        if far <= tolerance:
            return [chain[0], chain[-1]]
        return rdp(chain[:index + 1])[:-1] + rdp(chain[index:])

    first = 0
    second = max(range(len(points)), key=lambda i: hypot(points[i][0] - points[0][0],
                                                          points[i][1] - points[0][1]))
    left = rdp(points[first:second + 1])
    right = rdp(points[second:] + [points[0]])
    return left[:-1] + right[:-1]


def render_geometry(grid: GridMap) -> dict[str, Any]:
    """Describe the arena as clean vector shapes instead of pixels.

    The occupancy grid draws walls as one-cell outlines and pillars as rings, which
    looks torn when scaled. This extracts the floor outline (simplified convex
    polygon) and the pillars (circles) in world metres.
    """
    res = grid.resolution
    outside = _outside(grid)
    solid = grid.occupied | (grid.unknown & ~outside)
    floor = ~outside & ~solid

    # Floor outline: hull of the corners of the floor's boundary cells.
    padded = np.pad(floor, 1)
    edge = floor & ~(padded[:-2, 1:-1] & padded[2:, 1:-1] & padded[1:-1, :-2] & padded[1:-1, 2:])
    corners: list[tuple[float, float]] = []
    for r, c in zip(*np.where(edge)):
        for dr in (0, 1):
            for dc in (0, 1):
                corners.append((grid.origin_x + (c + dc) * res, grid.origin_y + (r + dr) * res))
    hull = simplify_polygon(convex_hull(corners), tolerance=res * 1.4)
    perimeter = sum(hypot(b[0] - a[0], b[1] - a[1]) for a, b in zip(hull, hull[1:] + hull[:1]))

    # Pillars: small solid blobs inside the arena; the biggest blob is the wall ring.
    blobs = sorted(_components(solid), key=len, reverse=True)
    wall_cells = len(blobs[0]) if blobs else 0
    pillars = []
    for cells in blobs[1:]:
        rows = [r for r, _ in cells]
        cols = [c for _, c in cells]
        height, width = max(rows) - min(rows) + 1, max(cols) - min(cols) + 1
        if len(cells) > 400:  # not a pillar
            continue
        # Centre of the blob's bounding box; the ring is one cell thick, so the
        # true radius is half a cell smaller than the box suggests.
        x = grid.origin_x + (min(cols) + max(cols) + 1) / 2.0 * res
        y = grid.origin_y + (min(rows) + max(rows) + 1) / 2.0 * res
        pillars.append({'x': round(float(x), 3), 'y': round(float(y), 3),
                        'r': round((max(height, width) - 1) * res / 2.0, 3)})
    thickness = wall_cells * res * res / perimeter if perimeter else 0.1
    xs = [p[0] for p in hull]
    ys = [p[1] for p in hull]
    margin = 0.4
    return {
        'floor': [[round(float(x), 3), round(float(y), 3)] for x, y in hull],
        'pillars': pillars,
        'wall': round(float(min(0.2, max(0.05, thickness))), 3),
        'bounds': {'xmin': round(float(min(xs)) - margin, 3), 'xmax': round(float(max(xs)) + margin, 3),
                   'ymin': round(float(min(ys)) - margin, 3), 'ymax': round(float(max(ys)) + margin, 3)},
        'resolution': float(res),
        'origin': [float(grid.origin_x), float(grid.origin_y)],
    }


# --- data store ---------------------------------------------------------------------


def value_runs(
    values: np.ndarray,
    mask: np.ndarray,
    *,
    step: float = 0.05,
) -> list[list[float]]:
    """Run-length encode selected grid values as ``[row, c0, c1, value]``.

    Values are bucketed before encoding.  The learner changes neighbouring cells
    by tiny floating-point amounts; sending every such cell separately made the
    dashboard payload both noisy and unnecessarily large.
    """
    if values.shape != mask.shape:
        raise ValueError('values and mask must have the same shape')
    quantized = np.rint(values / step) * step
    runs: list[list[float]] = []
    for row in np.where(mask.any(axis=1))[0]:
        line = quantized[row]
        selected = mask[row]
        col, count = 0, len(line)
        while col < count:
            if not selected[col]:
                col += 1
                continue
            end = col
            while end + 1 < count and selected[end + 1] and line[end + 1] == line[col]:
                end += 1
            runs.append([int(row), col, end, round(float(line[col]), 2)])
            col = end + 1
    return runs


def mask_runs(mask: np.ndarray) -> list[list[int]]:
    """Run-length encode a boolean grid as ``[row, c0, c1]``."""
    runs: list[list[int]] = []
    for row in np.where(mask.any(axis=1))[0]:
        line = mask[row]
        col, count = 0, len(line)
        while col < count:
            if not line[col]:
                col += 1
                continue
            end = col
            while end + 1 < count and line[end + 1]:
                end += 1
            runs.append([int(row), col, end])
            col = end + 1
    return runs


def terrain_runs(terrain: np.ndarray) -> list[list[float]]:
    """Compatibility encoder for non-neutral floor prices."""
    return value_runs(terrain, ~np.isclose(terrain, 1.0), step=0.001)


def costmap_layers(costmap) -> dict[str, Any]:
    """Build the compact display layers for the agent and planner maps.

    ``knowledge`` includes neutral cells because the UI must distinguish a
    measured ordinary floor from an unvisited cell.  Planner layers use their
    natural neutral background and therefore only carry non-zero deviations.
    """
    arena = ~_outside(costmap.grid)
    static_free = arena & ~costmap.static_blocked
    traversable = arena & ~costmap.blocked
    known = static_free & (costmap.last_seen > -1e8)
    terrain = costmap.terrain
    wall_cost = costmap.wall_cost
    total = terrain + wall_cost
    return {
        'version': int(costmap.version),
        'knowledge': value_runs(terrain, known),
        'terrain': value_runs(terrain, traversable & ~np.isclose(terrain, 1.0)),
        'wall_cost': value_runs(wall_cost, traversable & (wall_cost > 0.025)),
        'total': value_runs(total, traversable & ~np.isclose(total, 1.0)),
        'blocked': mask_runs(arena & costmap.blocked),
    }


class DashboardData:
    """Everything the page shows, updated by ROS callbacks, read by HTTP threads."""

    def __init__(self, base: tuple[float, float] = (-2.0, -0.5)) -> None:
        self.lock = threading.Lock()
        self.base = base
        self.pose: dict[str, float] | None = None
        self.state: dict[str, Any] = {}
        self.status: dict[str, Any] = {}
        self.score: dict[str, Any] = {}
        self.costmap: dict[str, Any] = {
            'version': -1,
            'knowledge': [],
            'terrain': [],
            'wall_cost': [],
            'total': [],
            'blocked': [],
        }
        self.truth: dict[str, Any] | None = None
        self.plan_text = ''
        self.events: deque[dict[str, Any]] = deque(maxlen=60)
        self.journal: deque[dict[str, Any]] = deque(maxlen=100)
        self.trail: deque[tuple[float, float]] = deque(maxlen=3000)
        self.collected_at: list[tuple[float, float]] = []

    def on_pose(self, x: float, y: float, yaw: float) -> None:
        with self.lock:
            self.pose = {'x': round(x, 3), 'y': round(y, 3), 'yaw': round(yaw, 3)}
            if not self.trail or hypot(x - self.trail[-1][0], y - self.trail[-1][1]) >= 0.04:
                self.trail.append((round(x, 3), round(y, 3)))

    def on_state(self, state: dict[str, Any]) -> None:
        with self.lock:
            self.state = state

    def on_status(self, status: dict[str, Any]) -> None:
        with self.lock:
            self.status = status

    def run_state(self) -> str:
        """Return the executor state without copying the full dashboard payload."""
        with self.lock:
            return str(self.status.get('state') or 'idle')

    def on_score(self, score: dict[str, Any]) -> None:
        with self.lock:
            self.score = score

    def on_costmap(self, costmap: dict[str, Any]) -> None:
        with self.lock:
            self.costmap = costmap

    def on_truth(self, truth: dict[str, Any]) -> None:
        with self.lock:
            self.truth = truth

    def on_event(self, event: dict[str, Any]) -> None:
        with self.lock:
            self.events.append(event)
            if event.get('event') == 'sample_collected' and self.pose:
                self.collected_at.append((self.pose['x'], self.pose['y']))

    def on_journal(self, entry: dict[str, Any]) -> None:
        with self.lock:
            self.journal.append(entry)

    def on_plan(self, text: str) -> None:
        """Remember the last plan and its explanation (shown as a decision)."""
        with self.lock:
            self.plan_text = text
        try:
            data = json.loads(text)
        except ValueError:
            return
        if not isinstance(data, dict) or not data.get('explanation'):
            return
        subgoals = data.get('subgoals')
        items = subgoals if isinstance(subgoals, list) else []
        source = str(data.get('source') or 'llm')
        kind = 'llm' if source == 'llm' else 'robot'
        search = next((item for item in items
                       if isinstance(item, dict)
                       and item.get('type') == 'search_around'), None)
        if search and all(isinstance(search.get(key), (int, float))
                          for key in ('x', 'y', 'radius')):
            action = ('Выбран район поиска' if kind == 'llm'
                      else 'Резервный выбор района' if source == 'fallback'
                      else 'Автоматический локальный поиск')
            title = (
                f'{action} '
                f'({float(search["x"]):.2f}; {float(search["y"]):.2f}), '
                f'радиус {float(search["radius"]):.2f}'
            )
        else:
            action = ('Резервный план' if source == 'fallback'
                      else 'План по запасу энергии' if source == 'budget'
                      else 'Выбран план')
            title = f'{action} {data.get("plan_id", "")}'.strip()
        steps = ' → '.join(
            str(item.get('type')) for item in items
            if isinstance(item, dict) and item.get('type')
        )
        detail = str(data['explanation'])
        if steps:
            detail += f'\nПодцели: {steps}'
        self.on_journal({
            'kind': kind,
            'title': title,
            'text': detail,
            't': self.state.get('t'),
        })

    def reset_run(self, *, reset_pose: bool = False) -> None:
        """Clear episode data, optionally restoring the Gazebo spawn pose."""
        with self.lock:
            if reset_pose:
                self.pose = {'x': self.base[0], 'y': self.base[1], 'yaw': 0.0}
            self.state = {}
            self.status = {}
            self.score = {}
            self.costmap = {
                'version': -1,
                'knowledge': [],
                'terrain': [],
                'wall_cost': [],
                'total': [],
                'blocked': [],
            }
            self.truth = None
            self.plan_text = ''
            self.events.clear()
            self.journal.clear()
            self.trail.clear()
            if self.pose is not None:
                self.trail.append((self.pose['x'], self.pose['y']))
            self.collected_at.clear()

    def snapshot(self) -> dict[str, Any]:
        with self.lock:
            return {
                'pose': self.pose,
                'state': self.state,
                'status': self.status,
                'score': self.score,
                'trail': [list(point) for point in self.trail],
                'events': list(self.events),
                'journal': list(self.journal),
                'costmap': self.costmap,
                'plan': self.plan_text,
                'collected_at': self.collected_at,
                'scenario': self.score.get('scenario'),
            }


def truth_from_scenario(scenario, at: float = 0.0) -> dict[str, Any]:
    """Hidden ground truth at simulation time ``at`` for the debug overlay."""
    soil_zones = {zone.id: asdict(zone) for zone in scenario.soil_zones}
    hazard_zones = [asdict(zone) for zone in scenario.hazard_zones]
    for event in scenario.events:
        if event.at > at:
            break
        if event.type == 'soil_change' and event.zone in soil_zones:
            soil_zones[event.zone]['cost_multiplier'] = event.cost_multiplier
        elif event.type == 'hazard_appear':
            hazard_zones.append(asdict(event.zone))
    return {
        'name': scenario.name,
        'at': float(at),
        'base': {'x': scenario.base_x, 'y': scenario.base_y},
        'samples': [asdict(s) for s in scenario.samples],
        'soil_zones': list(soil_zones.values()),
        'hazard_zones': hazard_zones,
        'events': [
            {'at': e.at, 'type': e.type} for e in scenario.events
        ],
    }


def preview_from_scenario(scenario) -> dict[str, Any]:
    """Full setup preview, including dynamic zones that appear later."""
    preview = truth_from_scenario(scenario, 0.0)
    events: list[dict[str, Any]] = []
    future_hazards: list[dict[str, Any]] = []
    for event in scenario.events:
        item: dict[str, Any] = {'at': event.at, 'type': event.type}
        if event.type == 'soil_change':
            item.update(zone=event.zone, cost_multiplier=event.cost_multiplier)
        elif event.type == 'hazard_appear':
            zone = asdict(event.zone)
            item['zone'] = zone
            future_hazards.append({**zone, 'appears_at': event.at})
        elif event.type == 'sensor_fault':
            item.update(noise_stddev=event.noise_stddev, duration=event.duration)
        events.append(item)
    preview.update(
        seed=scenario.seed,
        events=events,
        future_hazard_zones=future_hazards,
    )
    return preview


# --- HTTP ----------------------------------------------------------------------------------------


class DashboardServer:
    """Serve the page, the map image and a small JSON API."""

    def __init__(
        self,
        data: DashboardData,
        geometry: dict[str, Any],
        send_plan: Callable[[str], None],
        send_command: Callable[[str], None],
        send_scenario: Callable[[str], None] | None = None,
        preview_scenario: Callable[[str], dict[str, Any]] | None = None,
        *,
        set_navigation_backend: Callable[[str], None] | None = None,
        port: int = 8080,
        host: str = '0.0.0.0',
        log: Callable[[str], None] = lambda message: None,
    ) -> None:
        self.data = data
        self.geometry = geometry
        self.send_plan = send_plan
        self.send_command = send_command
        self.send_scenario = send_scenario
        self.preview_scenario = preview_scenario
        self.set_navigation_backend = set_navigation_backend
        self.log = log
        owner = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *args) -> None:  # keep the node log clean
                pass

            def _send(self, code: int, body: bytes, kind: str) -> None:
                self.send_response(code)
                self.send_header('Content-Type', kind)
                self.send_header('Content-Length', str(len(body)))
                self.send_header('Cache-Control', 'no-store')
                self.end_headers()
                self.wfile.write(body)

            def _json(self, payload: Any, code: int = 200) -> None:
                self._send(code, json.dumps(payload).encode(), 'application/json')

            def do_GET(self) -> None:  # noqa: N802 - http.server API
                path = self.path.split('?')[0]
                if path in ('/', '/index.html'):
                    page = web_dir() / 'index.html'
                    self._send(200, page.read_bytes(), 'text/html; charset=utf-8')
                elif path == '/api/geometry':
                    self._json(owner.geometry)
                elif path == '/api/state':
                    self._json(owner.data.snapshot())
                elif path == '/api/truth':
                    self._json(owner.data.truth or {})
                else:
                    self._send(404, b'not found', 'text/plain')

            def do_POST(self) -> None:  # noqa: N802 - http.server API
                length = int(self.headers.get('Content-Length') or 0)
                if length > MAX_BODY:
                    self._json({'ok': False, 'error': 'body too large'}, 413)
                    return
                try:
                    body = json.loads(self.rfile.read(length) or b'{}')
                except ValueError:
                    self._json({'ok': False, 'error': 'invalid JSON'}, 400)
                    return
                if not isinstance(body, dict):
                    self._json({'ok': False, 'error': 'JSON body must be an object'}, 400)
                    return
                path = self.path.split('?')[0]
                owner.log(
                    f'control request {path} {json.dumps(body, ensure_ascii=False)[:120]} '
                    f'from {self.client_address[0]} ({self.headers.get("User-Agent", "?")[:40]})'
                )
                if path == '/api/goto':
                    plan = owner.plan_for_click(body)
                elif path == '/api/plan':
                    plan = json.dumps(body)
                elif path == '/api/command':
                    command = body.get('cmd')
                    if command not in COMMANDS:
                        self._json({'ok': False, 'error': f'unknown command {command!r}'}, 400)
                        return
                    owner.send_command(command)
                    self._json({'ok': True})
                    return
                elif path == '/api/navigation':
                    backend = body.get('backend')
                    if not isinstance(backend, str) or backend not in NAVIGATION_BACKENDS:
                        self._json({
                            'ok': False,
                            'error': 'navigation backend must be custom or nav2',
                        }, 400)
                        return
                    if owner.set_navigation_backend is None:
                        self._json({'ok': False, 'error': 'navigation control unavailable'}, 503)
                        return
                    try:
                        owner.set_navigation_backend(backend)
                    except (OSError, RuntimeError, ValueError) as error:
                        self._json({'ok': False, 'error': str(error)}, 503)
                        return
                    # Only /agent/state confirms which backend is now active.
                    self._json({'ok': True, 'requested_backend': backend, 'pending': True}, 202)
                    return
                elif path == '/api/scenario':
                    scenario = body.get('scenario')
                    if not valid_scenario_name(scenario):
                        self._json({
                            'ok': False,
                            'error': f'unknown scenario {scenario!r}',
                        }, 400)
                        return
                    if owner.send_scenario is None:
                        self._json({'ok': False, 'error': 'scenario control unavailable'}, 503)
                        return
                    try:
                        owner.send_scenario(scenario)
                    except (OSError, RuntimeError, ValueError) as error:
                        self._json({'ok': False, 'error': str(error)}, 400)
                        return
                    self._json({'ok': True, 'scenario': scenario})
                    return
                elif path == '/api/scenario/preview':
                    scenario = body.get('scenario')
                    if not valid_scenario_name(scenario):
                        self._json({
                            'ok': False,
                            'error': f'unknown scenario {scenario!r}',
                        }, 400)
                        return
                    if owner.preview_scenario is None:
                        self._json({'ok': False, 'error': 'scenario preview unavailable'}, 503)
                        return
                    try:
                        preview = owner.preview_scenario(scenario)
                    except (OSError, RuntimeError, ValueError) as error:
                        self._json({'ok': False, 'error': str(error)}, 400)
                        return
                    self._json({'ok': True, 'scenario': scenario, 'preview': preview})
                    return
                else:
                    self._send(404, b'not found', 'text/plain')
                    return
                try:
                    parse_plan(plan)
                except PlanError as error:
                    self._json({'ok': False, 'error': str(error)}, 400)
                    return
                owner.send_plan(plan)
                self._json({'ok': True})

        self.httpd = ThreadingHTTPServer((host, port), Handler)
        self.thread = threading.Thread(target=self.httpd.serve_forever, daemon=True)

    @property
    def port(self) -> int:
        return self.httpd.server_address[1]

    @staticmethod
    def plan_for_click(body: dict[str, Any]) -> str:
        """Build the plan for a click: go there, or search there and collect."""
        x, y = body.get('x'), body.get('y')
        if not all(isinstance(v, (int, float)) and not isinstance(v, bool) and isfinite(v)
                   for v in (x, y)):
            subgoals = [{'type': 'goto', 'x': x, 'y': y}]  # parse_plan names the bad field
            return json.dumps({'plan_id': 'ui', 'subgoals': subgoals})
        if body.get('mode') == 'search':
            subgoals = [
                {'type': 'search_around', 'x': x, 'y': y, 'radius': 0.8},
                {'type': 'collect'},
            ]
        else:
            subgoals = [{'type': 'goto', 'x': x, 'y': y}]
        return json.dumps({'plan_id': 'ui', 'subgoals': subgoals})

    def start(self) -> None:
        self.thread.start()

    def stop(self) -> None:
        self.httpd.shutdown()
        self.httpd.server_close()
