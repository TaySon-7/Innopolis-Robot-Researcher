"""Scenario description loaded from YAML (see docs/INTERFACES.md)."""

from __future__ import annotations

from dataclasses import dataclass
from dataclasses import field
from math import hypot
from pathlib import Path
from typing import Any

import yaml


@dataclass
class Zone:
    """A circular or rectangular area of the arena in world coordinates."""

    id: str
    shape: str
    x: float = 0.0
    y: float = 0.0
    radius: float = 0.0
    x_min: float = 0.0
    y_min: float = 0.0
    x_max: float = 0.0
    y_max: float = 0.0
    cost_multiplier: float = 1.0
    penalty: float = 0.0

    def contains(self, x: float, y: float) -> bool:
        """Return whether the world point lies inside the zone."""
        if self.shape == 'circle':
            return hypot(x - self.x, y - self.y) <= self.radius
        return self.x_min <= x <= self.x_max and self.y_min <= y <= self.y_max


@dataclass
class ScenarioSample:
    """A hidden sample position."""

    id: str
    x: float
    y: float


@dataclass
class ScenarioEvent:
    """A silent change of the environment at a given simulation time."""

    at: float
    type: str
    zone: Any = None
    cost_multiplier: float = 1.0
    noise_stddev: float = 0.0
    duration: float = 0.0


@dataclass
class Scenario:
    """Everything the judge needs to run one episode."""

    name: str
    seed: int
    base_x: float
    base_y: float
    initial_battery: float
    battery_cost_per_meter: float
    collection_radius: float
    sensor_range: float
    sensor_noise_stddev: float
    samples: list[ScenarioSample] = field(default_factory=list)
    soil_zones: list[Zone] = field(default_factory=list)
    hazard_zones: list[Zone] = field(default_factory=list)
    events: list[ScenarioEvent] = field(default_factory=list)


EVENT_TYPES = ('soil_change', 'hazard_appear', 'sensor_fault')


def zone_from_dict(data: dict[str, Any]) -> Zone:
    """Build a zone from its YAML mapping and validate its shape."""
    shape = data.get('shape')
    if shape not in ('circle', 'rect'):
        raise ValueError(f'zone {data.get("id")!r}: unknown shape {shape!r}')
    zone = Zone(
        id=str(data['id']),
        shape=shape,
        x=float(data.get('x', 0.0)),
        y=float(data.get('y', 0.0)),
        radius=float(data.get('radius', 0.0)),
        x_min=float(data.get('x_min', 0.0)),
        y_min=float(data.get('y_min', 0.0)),
        x_max=float(data.get('x_max', 0.0)),
        y_max=float(data.get('y_max', 0.0)),
        cost_multiplier=float(data.get('cost_multiplier', 1.0)),
        penalty=float(data.get('penalty', 0.0)),
    )
    if shape == 'circle' and zone.radius <= 0.0:
        raise ValueError(f'zone {zone.id!r}: circle needs a positive radius')
    if shape == 'rect' and (zone.x_max <= zone.x_min or zone.y_max <= zone.y_min):
        raise ValueError(f'zone {zone.id!r}: rect has an empty area')
    return zone


def scenario_from_dict(data: dict[str, Any]) -> Scenario:
    """Build and validate a scenario from a parsed YAML mapping."""
    battery = data.get('battery', {})
    sensor = data.get('sensor', {})
    base = data['base']
    scenario = Scenario(
        name=str(data.get('name', 'unnamed')),
        seed=int(data.get('seed', 0)),
        base_x=float(base['x']),
        base_y=float(base['y']),
        initial_battery=float(battery.get('initial', 60.0)),
        battery_cost_per_meter=float(battery.get('cost_per_meter', 1.0)),
        collection_radius=float(data.get('collection_radius', 0.30)),
        sensor_range=float(sensor.get('range', 1.5)),
        sensor_noise_stddev=float(sensor.get('noise_stddev', 0.01)),
        samples=[
            ScenarioSample(str(item['id']), float(item['x']), float(item['y']))
            for item in data.get('samples', [])
        ],
        soil_zones=[zone_from_dict(item) for item in data.get('soil_zones', [])],
        hazard_zones=[zone_from_dict(item) for item in data.get('hazard_zones', [])],
    )
    soil_ids = {zone.id for zone in scenario.soil_zones}
    for item in sorted(data.get('events', []), key=lambda entry: entry['at']):
        kind = item['type']
        if kind not in EVENT_TYPES:
            raise ValueError(f'unknown event type {kind!r}')
        event = ScenarioEvent(at=float(item['at']), type=kind)
        if kind == 'soil_change':
            if item['zone'] not in soil_ids:
                raise ValueError(f'soil_change refers to unknown zone {item["zone"]!r}')
            event.zone = str(item['zone'])
            event.cost_multiplier = float(item['cost_multiplier'])
        elif kind == 'hazard_appear':
            event.zone = zone_from_dict(item['zone'])
        else:
            event.noise_stddev = float(item['noise_stddev'])
            event.duration = float(item.get('duration', 0.0))
        scenario.events.append(event)
    return scenario


def _zone_to_dict(zone: Zone, extra: dict[str, Any]) -> dict[str, Any]:
    if zone.shape == 'circle':
        data = {'id': zone.id, 'shape': 'circle', 'x': zone.x, 'y': zone.y, 'radius': zone.radius}
    else:
        data = {'id': zone.id, 'shape': 'rect', 'x_min': zone.x_min, 'y_min': zone.y_min,
                'x_max': zone.x_max, 'y_max': zone.y_max}
    data.update(extra)
    return data


def scenario_to_dict(scenario: Scenario) -> dict[str, Any]:
    """Return the YAML mapping of a scenario (the inverse of scenario_from_dict)."""
    events = []
    for event in scenario.events:
        item: dict[str, Any] = {'at': event.at, 'type': event.type}
        if event.type == 'soil_change':
            item.update(zone=event.zone, cost_multiplier=event.cost_multiplier)
        elif event.type == 'hazard_appear':
            item['zone'] = _zone_to_dict(event.zone, {'penalty': event.zone.penalty})
        else:
            item.update(noise_stddev=event.noise_stddev, duration=event.duration)
        events.append(item)
    return {
        'name': scenario.name,
        'seed': scenario.seed,
        'base': {'x': scenario.base_x, 'y': scenario.base_y},
        'battery': {'initial': scenario.initial_battery,
                    'cost_per_meter': scenario.battery_cost_per_meter},
        'collection_radius': scenario.collection_radius,
        'sensor': {'range': scenario.sensor_range, 'noise_stddev': scenario.sensor_noise_stddev},
        'samples': [{'id': s.id, 'x': s.x, 'y': s.y} for s in scenario.samples],
        'soil_zones': [_zone_to_dict(z, {'cost_multiplier': z.cost_multiplier})
                       for z in scenario.soil_zones],
        'hazard_zones': [_zone_to_dict(z, {'penalty': z.penalty}) for z in scenario.hazard_zones],
        'events': events,
    }


def scenario_path(name: str) -> Path:
    """Return the file of a bundled scenario: installed share dir, else source tree."""
    try:
        from ament_index_python.packages import get_package_share_directory
        installed = Path(get_package_share_directory('did_judge')) / 'scenarios' / f'{name}.yaml'
        if installed.exists():
            return installed
    except Exception:  # noqa: BLE001 - plain Python run without ROS installed
        pass
    return Path(__file__).resolve().parent.parent / 'scenarios' / f'{name}.yaml'


def load_scenario(path: str | Path) -> Scenario:
    """Read a scenario YAML file."""
    with open(path, encoding='utf-8') as handle:
        return scenario_from_dict(yaml.safe_load(handle))
