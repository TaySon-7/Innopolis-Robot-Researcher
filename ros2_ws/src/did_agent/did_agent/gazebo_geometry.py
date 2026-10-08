"""Read the static 2-D arena geometry from Gazebo's live scene graph."""

from __future__ import annotations

from math import atan2
from math import hypot
from pathlib import Path
import xml.etree.ElementTree as ET
from typing import Any
from typing import Callable

from did_agent.dashboard_core import convex_hull

Pose2 = tuple[float, float, float]


def _pose2(message) -> Pose2:
    position = message.position
    orientation = message.orientation
    yaw = atan2(
        2.0 * (orientation.w * orientation.z + orientation.x * orientation.y),
        1.0 - 2.0 * (orientation.y * orientation.y + orientation.z * orientation.z),
    )
    return float(position.x), float(position.y), yaw


def _compose(parent: Pose2, child: Pose2) -> Pose2:
    from math import cos
    from math import sin

    px, py, pa = parent
    x, y, angle = child
    return (
        px + x * cos(pa) - y * sin(pa),
        py + x * sin(pa) + y * cos(pa),
        pa + angle,
    )


def _transform(points: list[tuple[float, float]], pose: Pose2) -> list[tuple[float, float]]:
    from math import cos
    from math import sin

    x, y, angle = pose
    ca, sa = cos(angle), sin(angle)
    return [(x + px * ca - py * sa, y + px * sa + py * ca) for px, py in points]


def _polygon_area(points: list[tuple[float, float]]) -> float:
    return abs(sum(a[0] * b[1] - b[0] * a[1]
                   for a, b in zip(points, points[1:] + points[:1]))) / 2.0


def _perimeter(points: list[tuple[float, float]]) -> float:
    return sum(hypot(b[0] - a[0], b[1] - a[1])
               for a, b in zip(points, points[1:] + points[:1]))


def _position_values(path: Path) -> tuple[list[tuple[float, float, float]], float]:
    """Return COLLADA position triples and its unit-to-metre coefficient."""
    root = ET.parse(path).getroot()
    namespace = root.tag.partition('}')[0].lstrip('{')
    prefix = f'{{{namespace}}}' if namespace else ''
    unit = root.find(f'.//{prefix}asset/{prefix}unit')
    metre = float(unit.get('meter', '1')) if unit is not None else 1.0
    vertices = root.find(f'.//{prefix}vertices')
    if vertices is None:
        raise ValueError(f'{path}: COLLADA has no vertices')
    position = next((item for item in vertices.findall(f'{prefix}input')
                     if item.get('semantic') == 'POSITION'), None)
    if position is None:
        raise ValueError(f'{path}: COLLADA vertices have no POSITION source')
    source_id = position.get('source', '').lstrip('#')
    source = root.find(f".//{prefix}source[@id='{source_id}']")
    if source is None:
        raise ValueError(f'{path}: missing position source {source_id!r}')
    array = source.find(f'{prefix}float_array')
    accessor = source.find(f'.//{prefix}accessor')
    if array is None or not array.text:
        raise ValueError(f'{path}: empty position source')
    stride = int(accessor.get('stride', '3')) if accessor is not None else 3
    raw = [float(value) for value in array.text.split()]
    return [tuple(raw[index:index + 3]) for index in range(0, len(raw), stride)], metre


def _planar_axes(vertices: list[tuple[float, float, float]]) -> tuple[int, int]:
    """Find a mesh's planar dimensions without changing their coordinate order."""
    spans = [max(point[axis] for point in vertices) - min(point[axis] for point in vertices)
             for axis in range(3)]
    return tuple(sorted(sorted(range(3), key=lambda axis: spans[axis], reverse=True)[:2]))


def _wall_shells(path: Path, scale: tuple[float, float, float]) -> tuple[
    list[tuple[float, float]], list[tuple[float, float]]
]:
    """Extract the inner and outer planar shells of a wall mesh."""
    vertices, metre = _position_values(path)
    # Find the two planar dimensions, but keep their original X/Y ordering.
    # Ordering by span would swap X and Y for wall.dae (Y is slightly wider),
    # effectively rotating the arena before its SDF pose is applied.
    axes = _planar_axes(vertices)
    points = sorted({
        (round(point[axes[0]] * metre * scale[axes[0]], 7),
         round(point[axes[1]] * metre * scale[axes[1]], 7))
        for point in vertices
    })
    centre = (
        (min(x for x, _ in points) + max(x for x, _ in points)) / 2.0,
        (min(y for _, y in points) + max(y for _, y in points)) / 2.0,
    )
    by_radius = sorted((hypot(x - centre[0], y - centre[1]), (x, y)) for x, y in points)
    clusters: list[list[tuple[float, tuple[float, float]]]] = []
    for radius, point in by_radius:
        if not clusters:
            clusters.append([(radius, point)])
            continue
        mean = sum(item[0] for item in clusters[-1]) / len(clusters[-1])
        if abs(radius - mean) <= max(1e-6, mean * 0.02):
            clusters[-1].append((radius, point))
        else:
            clusters.append([(radius, point)])
    shells = [convex_hull([point for _, point in cluster])
              for cluster in clusters if len(cluster) >= 3]
    if len(shells) < 2:
        raise ValueError(f'{path}: wall mesh needs inner and outer shells')
    return shells[0], shells[-1]


def _has(message, field: str) -> bool:
    try:
        return bool(message.HasField(field))
    except (AttributeError, ValueError):
        return getattr(message, field, None) is not None


def _walk_models(models, parent: Pose2 = (0.0, 0.0, 0.0)):
    for model in models:
        model_pose = _compose(parent, _pose2(model.pose))
        yield model, model_pose
        yield from _walk_models(model.model, model_pose)


def geometry_from_scene(scene, fallback: dict[str, Any]) -> dict[str, Any]:
    """Normalize a ``gz.msgs.Scene`` into the dashboard's 2-D geometry."""
    wall: tuple[list[tuple[float, float]], list[tuple[float, float]], str] | None = None
    pillars: list[dict[str, float]] = []
    for model, model_pose in _walk_models(scene.model):
        # The robot also contains visuals.  Only the static arena model is geometry.
        if model.name in ('burger', 'ground_plane'):
            continue
        for link in model.link:
            link_pose = _compose(model_pose, _pose2(link.pose))
            for visual in link.visual:
                pose = _compose(link_pose, _pose2(visual.pose))
                geometry = visual.geometry
                if _has(geometry, 'cylinder'):
                    radius = float(geometry.cylinder.radius)
                    if 0.05 <= radius <= 0.5:
                        x, y, _ = pose
                        pillars.append({'x': round(x, 3), 'y': round(y, 3),
                                        'r': round(radius, 3)})
                elif _has(geometry, 'mesh'):
                    filename = str(geometry.mesh.filename)
                    if Path(filename).name != 'wall.dae':
                        continue
                    mesh_scale = geometry.mesh.scale
                    scale = tuple(float(value) or 1.0
                                  for value in (mesh_scale.x, mesh_scale.y, mesh_scale.z))
                    inner, outer = _wall_shells(Path(filename), scale)
                    wall = (_transform(inner, pose), _transform(outer, pose), filename)
    if wall is None:
        raise ValueError('Gazebo scene has no wall.dae visual')

    inner, outer, filename = wall
    floor = convex_hull(inner)
    outer = convex_hull(outer)
    average_perimeter = (_perimeter(floor) + _perimeter(outer)) / 2.0
    thickness = ((_polygon_area(outer) - _polygon_area(floor)) / average_perimeter
                 if average_perimeter else fallback.get('wall', 0.1))
    unique_pillars = {
        (pillar['x'], pillar['y'], pillar['r']): pillar for pillar in pillars
    }
    xs, ys = [point[0] for point in outer], [point[1] for point in outer]
    margin = 0.35
    return {
        **fallback,
        'source': 'gazebo_scene',
        'scene_service': '/world/default/scene/info',
        'wall_mesh': Path(filename).name,
        'floor': [[round(x, 4), round(y, 4)] for x, y in floor],
        'pillars': sorted(unique_pillars.values(), key=lambda item: (item['x'], item['y'])),
        'wall': round(float(thickness), 4),
        'bounds': {
            'xmin': round(min(xs) - margin, 3),
            'xmax': round(max(xs) + margin, 3),
            'ymin': round(min(ys) - margin, 3),
            'ymax': round(max(ys) + margin, 3),
        },
    }


def request_gazebo_geometry(
    fallback: dict[str, Any],
    *,
    timeout: int = 1500,
    log: Callable[[str], None] | None = None,
) -> dict[str, Any]:
    """Request the live scene once; return the map geometry on any failure."""
    try:
        from gz.msgs10.empty_pb2 import Empty
        from gz.msgs10.scene_pb2 import Scene
        from gz.transport13 import Node

        executed, scene = Node().request(
            '/world/default/scene/info', Empty(), Empty, Scene, timeout,
        )
        if not executed:
            raise RuntimeError('scene service timed out')
        geometry = geometry_from_scene(scene, fallback)
        if len(geometry['pillars']) != 9:
            raise ValueError(f"expected 9 pillars, got {len(geometry['pillars'])}")
        return geometry
    except Exception as error:  # noqa: BLE001 - Gazebo is an optional runtime source
        if log is not None:
            log(f'Gazebo scene geometry unavailable, using navigation map: {error}')
        return {**fallback, 'source': 'navigation_map'}
