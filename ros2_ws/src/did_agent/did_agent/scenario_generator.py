"""Generate valid, reproducible scenarios on the real map.

    ros2 run did_agent generate_scenario --difficulty hard --seed 7 --out hard7.yaml
    ros2 run did_agent generate_scenario --check my_scenario.yaml

A scenario is valid when every sample and zone lies on free floor away from the
walls, every sample can be reached from the base (also after a hazard appears),
nothing hostile covers the base, and driving to all samples and home fits the
battery with room to search.
"""

from __future__ import annotations

import argparse
from collections import deque
from math import hypot
from pathlib import Path
import random
import sys
from typing import Any

import numpy as np
import yaml

from did_judge.scenario import Scenario
from did_judge.scenario import ScenarioEvent
from did_judge.scenario import ScenarioSample
from did_judge.scenario import Zone
from did_judge.scenario import load_scenario
from did_judge.scenario import scenario_to_dict

from did_agent.costmap import CostMap
from did_agent.planner import plan_waypoints

# difficulty -> (samples, soil zones, silent events)
SPECS = {'easy': (3, 1, False), 'medium': (5, 3, False), 'hard': (7, 4, True)}
BASE = (-2.0, -0.5)
SAMPLE_CLEARANCE = 0.35       # metres from walls and pillars
MIN_SAMPLE_SPACING = 0.7
MIN_BASE_DISTANCE = 1.0
TOUR_BUDGET_SHARE = 0.45      # the tour may use this share of the battery
HAZARD_RADIUS = 0.3


class ScenarioError(RuntimeError):
    """No valid scenario could be generated."""


class World:
    """The map facts the generator needs: free space, reachability, path costs."""

    def __init__(self, costmap: CostMap | None = None) -> None:
        self.costmap = costmap or CostMap()
        self.start = self.costmap.nearest_free(*self.costmap.world_to_cell(*BASE), 0.5)
        self.reachable = self._reach(self.costmap.blocked)

    def _reach(self, blocked: np.ndarray) -> np.ndarray:
        seen = np.zeros_like(blocked)
        if blocked[self.start]:
            return seen
        rows, cols = blocked.shape
        queue = deque([self.start])
        seen[self.start] = True
        while queue:
            r, c = queue.popleft()
            for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1), (1, 1), (1, -1), (-1, 1), (-1, -1)):
                a, b = r + dr, c + dc
                if 0 <= a < rows and 0 <= b < cols and not blocked[a, b] and not seen[a, b]:
                    seen[a, b] = True
                    queue.append((a, b))
        return seen

    def clearance(self, x: float, y: float) -> float:
        row, col = self.costmap.world_to_cell(x, y)
        if not self.costmap.grid.in_bounds(row, col):
            return 0.0
        return float(self.costmap.wall_distance[row, col])

    def reachable_at(self, x: float, y: float, mask: np.ndarray | None = None) -> bool:
        row, col = self.costmap.world_to_cell(x, y)
        mask = self.reachable if mask is None else mask
        return self.costmap.grid.in_bounds(row, col) and bool(mask[row, col])

    def candidates(self) -> list[tuple[float, float]]:
        """World points of cells that are reachable and far enough from walls."""
        limit = SAMPLE_CLEARANCE / self.costmap.resolution
        rows, cols = np.where(
            self.reachable & (self.costmap.wall_distance >= SAMPLE_CLEARANCE)
        )
        del limit
        return [self.costmap.cell_to_world(int(r), int(c)) for r, c in zip(rows, cols)]

    def with_truth(self, scenario: Scenario) -> CostMap:
        """A fresh cost map that knows the real floor prices of a scenario."""
        costmap = CostMap(self.costmap.grid)
        for zone in scenario.soil_zones:
            costmap.update(_region(zone), zone.cost_multiplier)
        return costmap


def _region(zone: Zone) -> dict[str, Any]:
    if zone.shape == 'circle':
        return {'circle': {'x': zone.x, 'y': zone.y, 'r': zone.radius}}
    return {'rect': {'x_min': zone.x_min, 'y_min': zone.y_min,
                     'x_max': zone.x_max, 'y_max': zone.y_max}}


def _centre(zone: Zone) -> tuple[float, float]:
    if zone.shape == 'circle':
        return zone.x, zone.y
    return (zone.x_min + zone.x_max) / 2.0, (zone.y_min + zone.y_max) / 2.0


def _extent(zone: Zone) -> float:
    if zone.shape == 'circle':
        return zone.radius
    return hypot(zone.x_max - zone.x_min, zone.y_max - zone.y_min) / 2.0


def tour_cost(scenario: Scenario, world: World) -> float:
    """Battery for a nearest-neighbour tour: base, every sample, base again."""
    costmap = world.with_truth(scenario)
    here = (scenario.base_x, scenario.base_y)
    pending = [(s.x, s.y) for s in scenario.samples]
    total = 0.0
    while pending:
        pending.sort(key=lambda p: hypot(p[0] - here[0], p[1] - here[1]))
        target = pending.pop(0)
        route = plan_waypoints(costmap, here, target)
        if route is None:
            return float('inf')
        total += costmap.energy_cost([here, *route])
        here = target
    route = plan_waypoints(costmap, here, (scenario.base_x, scenario.base_y))
    if route is None:
        return float('inf')
    return total + costmap.energy_cost([here, *route])


def validate(scenario: Scenario, world: World) -> list[str]:
    """Return a list of problems; an empty list means the scenario is valid."""
    problems: list[str] = []
    base = (scenario.base_x, scenario.base_y)
    if not world.reachable_at(*base):
        problems.append('base is not on reachable free floor')
    ids = [s.id for s in scenario.samples] + [z.id for z in scenario.soil_zones] \
        + [z.id for z in scenario.hazard_zones]
    ids += [e.zone.id for e in scenario.events if e.type == 'hazard_appear']
    if len(ids) != len(set(ids)):
        problems.append('duplicate ids')
    for sample in scenario.samples:
        if not world.reachable_at(sample.x, sample.y):
            problems.append(f'sample {sample.id} is not reachable from the base')
        elif world.clearance(sample.x, sample.y) < 0.3:
            problems.append(f'sample {sample.id} is too close to a wall')
    for first in range(len(scenario.samples)):
        for second in range(first + 1, len(scenario.samples)):
            a, b = scenario.samples[first], scenario.samples[second]
            if hypot(a.x - b.x, a.y - b.y) < 0.5:
                problems.append(f'samples {a.id} and {b.id} are too close to each other')
    hostile = list(scenario.hazard_zones) + [
        e.zone for e in scenario.events if e.type == 'hazard_appear'
    ]
    for zone in scenario.soil_zones + hostile:
        cx, cy = _centre(zone)
        if not world.reachable_at(cx, cy):
            problems.append(f'zone {zone.id} is not on free floor')
    for zone in hostile:
        if zone.contains(*base):
            problems.append(f'hazard {zone.id} covers the base')
        for sample in scenario.samples:
            if zone.contains(sample.x, sample.y):
                problems.append(f'hazard {zone.id} covers sample {sample.id}')
        blocked = world.costmap.blocked.copy()
        blocked |= world.costmap.cells_in_region(_region_circle(zone))
        mask = world._reach(blocked)
        for sample in scenario.samples:
            if not world.reachable_at(sample.x, sample.y, mask):
                problems.append(f'hazard {zone.id} cuts sample {sample.id} off')
    if not problems:
        cost = tour_cost(scenario, world)
        limit = scenario.initial_battery * TOUR_BUDGET_SHARE
        if cost > limit:
            problems.append(f'tour costs {cost:.1f}, more than {limit:.1f} of the battery')
    return problems


def _region_circle(zone: Zone) -> dict[str, Any]:
    cx, cy = _centre(zone)
    return {'circle': {'x': cx, 'y': cy, 'r': _extent(zone) + 0.1}}


def _pick_samples(rng: random.Random, world: World, candidates, count: int):
    chosen: list[tuple[float, float]] = []
    for _ in range(2000):
        x, y = rng.choice(candidates)
        if hypot(x - BASE[0], y - BASE[1]) < MIN_BASE_DISTANCE:
            continue
        if all(hypot(x - a, y - b) >= MIN_SAMPLE_SPACING for a, b in chosen):
            chosen.append((round(x, 2), round(y, 2)))
            if len(chosen) == count:
                return chosen
    return None


def _make_soil(rng: random.Random, world: World, candidates, index: int, taken):
    for _ in range(300):
        x, y = rng.choice(candidates)
        if hypot(x - BASE[0], y - BASE[1]) < 0.9:
            continue
        price = round(rng.uniform(1.8, 4.5) * 2) / 2.0
        if rng.random() < 0.65:
            zone = Zone(id=f'z{index}', shape='circle', x=round(x, 2), y=round(y, 2),
                        radius=round(rng.uniform(0.3, 0.5), 2), cost_multiplier=price)
        else:
            w, h = rng.uniform(0.5, 0.9), rng.uniform(0.4, 0.8)
            zone = Zone(id=f'z{index}', shape='rect', x_min=round(x - w / 2, 2),
                        y_min=round(y - h / 2, 2), x_max=round(x + w / 2, 2),
                        y_max=round(y + h / 2, 2), cost_multiplier=price)
        mask = world.costmap.cells_in_region(_region(zone))
        if mask.sum() < 20 or (mask & ~world.reachable).sum() > 0.3 * mask.sum():
            continue
        if any(hypot(x - tx, y - ty) < _extent(zone) + te + 0.1 for tx, ty, te in taken):
            continue
        taken.append((x, y, _extent(zone)))
        return zone
    return None


def _make_events(rng: random.Random, world: World, scenario: Scenario, candidates):
    events = [
        ScenarioEvent(at=float(rng.randrange(60, 121, 10)), type='soil_change',
                      zone=rng.choice(scenario.soil_zones).id,
                      cost_multiplier=round(rng.uniform(3.0, 5.0) * 2) / 2.0),
    ]
    for _ in range(400):
        x, y = rng.choice(candidates)
        zone = Zone(id='h1', shape='circle', x=round(x, 2), y=round(y, 2),
                    radius=HAZARD_RADIUS, penalty=5.0)
        if hypot(x - BASE[0], y - BASE[1]) < 1.2:
            continue
        if any(hypot(x - s.x, y - s.y) < HAZARD_RADIUS + 0.45 for s in scenario.samples):
            continue
        events.append(ScenarioEvent(at=float(rng.randrange(120, 181, 10)),
                                    type='hazard_appear', zone=zone))
        break
    events.append(ScenarioEvent(at=float(rng.randrange(180, 241, 10)), type='sensor_fault',
                                noise_stddev=0.15, duration=float(rng.randrange(45, 76, 5))))
    return sorted(events, key=lambda e: e.at)


def generate(
    difficulty: str,
    seed: int,
    world: World | None = None,
    *,
    samples: int | None = None,
    soils: int | None = None,
    battery: float = 60.0,
) -> Scenario:
    """Build a valid scenario; the same arguments always give the same result."""
    if difficulty not in SPECS:
        raise ScenarioError(f'unknown difficulty {difficulty!r} (use {", ".join(SPECS)})')
    world = world or World()
    sample_count, soil_count, silent_events = SPECS[difficulty]
    sample_count = sample_count if samples is None else samples
    soil_count = soil_count if soil_count is None or soils is None else soils
    rng = random.Random(f'{difficulty}:{seed}')
    candidates = world.candidates()
    last: list[str] = []
    for _ in range(300):
        points = _pick_samples(rng, world, candidates, sample_count)
        if points is None:
            continue
        taken: list[tuple[float, float, float]] = []
        zones = [_make_soil(rng, world, candidates, i + 1, taken) for i in range(soil_count)]
        if any(zone is None for zone in zones):
            continue
        scenario = Scenario(
            name=f'{difficulty}@{seed}', seed=seed, base_x=BASE[0], base_y=BASE[1],
            initial_battery=battery, battery_cost_per_meter=1.0, collection_radius=0.30,
            sensor_range=1.5, sensor_noise_stddev={'easy': 0.01}.get(difficulty, 0.02),
            samples=[ScenarioSample(f's{i + 1}', x, y) for i, (x, y) in enumerate(points)],
            soil_zones=zones,
        )
        if silent_events and zones:
            scenario.events = _make_events(rng, world, scenario, candidates)
        last = validate(scenario, world)
        if not last:
            return scenario
    raise ScenarioError(f'no valid {difficulty} scenario found for seed {seed}: {last}')


def load_named(name: str, world: World | None = None) -> Scenario:
    """Load a bundled scenario ('hard') or generate one ('hard@7')."""
    from did_judge.scenario import scenario_path
    if '@' in name:
        difficulty, _, seed = name.partition('@')
        if not seed.isdigit():
            raise ScenarioError(f'bad scenario name {name!r}, expected e.g. hard@7')
        return generate(difficulty, int(seed), world)
    return load_scenario(scenario_path(name))


def to_yaml(scenario: Scenario) -> str:
    """Render a scenario as YAML text."""
    return yaml.safe_dump(scenario_to_dict(scenario), sort_keys=False, allow_unicode=True)


def main(args=None) -> None:
    """Command-line entry point."""
    parser = argparse.ArgumentParser(description='Generate or check a scenario.')
    parser.add_argument('--difficulty', choices=list(SPECS), default='medium')
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--samples', type=int, help='override the number of samples')
    parser.add_argument('--soils', type=int, help='override the number of soil zones')
    parser.add_argument('--battery', type=float, default=60.0)
    parser.add_argument('--out', help='write YAML here instead of stdout')
    parser.add_argument('--check', metavar='FILE', help='validate an existing scenario file')
    options = parser.parse_args(args)

    world = World()
    if options.check:
        problems = validate(load_scenario(options.check), world)
        print('OK' if not problems else '\n'.join(problems))
        raise SystemExit(1 if problems else 0)
    try:
        scenario = generate(options.difficulty, options.seed, world, samples=options.samples,
                            soils=options.soils, battery=options.battery)
    except ScenarioError as error:
        print(error, file=sys.stderr)
        raise SystemExit(1)
    text = to_yaml(scenario)
    if options.out:
        Path(options.out).parent.mkdir(parents=True, exist_ok=True)
        Path(options.out).write_text(text, encoding='utf-8')
        print(f'wrote {options.out}: {scenario.name}, {len(scenario.samples)} samples, '
              f'{len(scenario.soil_zones)} soil zones, {len(scenario.events)} events')
    else:
        print(text, end='')


if __name__ == '__main__':
    main()
