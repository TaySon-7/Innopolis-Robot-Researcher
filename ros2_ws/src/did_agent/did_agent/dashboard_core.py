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
import threading
from typing import Any
from typing import Callable

import numpy as np

from did_agent.grid import GridMap
from did_agent.plan import PlanError
from did_agent.plan import parse_plan

WEB_DIR_SOURCE = Path(__file__).resolve().parent.parent / 'web'
COMMANDS = ('auto', 'stop')
MAX_BODY = 20_000


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


def terrain_runs(terrain: np.ndarray) -> list[list[float]]:
    """Run-length encode the cells whose floor price is not 1: [row, c0, c1, value]."""
    runs: list[list[float]] = []
    for row in np.where((terrain != 1.0).any(axis=1))[0]:
        line = terrain[row]
        col, count = 0, len(line)
        while col < count:
            if line[col] == 1.0:
                col += 1
                continue
            end = col
            while end + 1 < count and line[end + 1] == line[col]:
                end += 1
            runs.append([int(row), col, end, round(float(line[col]), 3)])
            col = end + 1
    return runs


class DashboardData:
    """Everything the page shows, updated by ROS callbacks, read by HTTP threads."""

    def __init__(self, base: tuple[float, float] = (-2.0, -0.5)) -> None:
        self.lock = threading.Lock()
        self.base = base
        self.pose: dict[str, float] | None = None
        self.state: dict[str, Any] = {}
        self.status: dict[str, Any] = {}
        self.score: dict[str, Any] = {}
        self.costmap: dict[str, Any] = {'version': -1, 'runs': []}
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

    def on_score(self, score: dict[str, Any]) -> None:
        with self.lock:
            self.score = score

    def on_costmap(self, costmap: dict[str, Any]) -> None:
        with self.lock:
            self.costmap = costmap

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
        if isinstance(data, dict) and data.get('explanation'):
            self.on_journal({
                'kind': 'decision',
                'title': f"План {data.get('plan_id', '')}".strip(),
                'text': str(data['explanation']),
                't': self.state.get('t'),
            })

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


def truth_from_scenario(scenario) -> dict[str, Any]:
    """Hidden ground truth of a scenario, for the optional overlay."""
    return {
        'name': scenario.name,
        'base': {'x': scenario.base_x, 'y': scenario.base_y},
        'samples': [asdict(s) for s in scenario.samples],
        'soil_zones': [asdict(z) for z in scenario.soil_zones],
        'hazard_zones': [asdict(z) for z in scenario.hazard_zones],
        'events': [
            {'at': e.at, 'type': e.type} for e in scenario.events
        ],
    }


# --- HTTP ----------------------------------------------------------------------------------------


class DashboardServer:
    """Serve the page, the map image and a small JSON API."""

    def __init__(
        self,
        data: DashboardData,
        geometry: dict[str, Any],
        send_plan: Callable[[str], None],
        send_command: Callable[[str], None],
        *,
        port: int = 8080,
        host: str = '0.0.0.0',
        log: Callable[[str], None] = lambda message: None,
    ) -> None:
        self.data = data
        self.geometry = geometry
        self.send_plan = send_plan
        self.send_command = send_command
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
