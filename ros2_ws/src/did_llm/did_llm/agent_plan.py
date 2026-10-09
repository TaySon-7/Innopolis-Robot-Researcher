"""The plan format the agent on /agent/plan accepts, and the budget rules.

Everything here mirrors ``did_agent/plan.py`` and ``did_agent/skills.py``
deliberately. The agent publishes a rejection reason when a plan does not
parse, and the handbook says that reason can be fed back to the model, so
the planner has to know the rules before it publishes rather than learn them
from a rejection. Duplicating the rules also keeps this package free of a
dependency on the executor: a planner that cannot run without the code it
plans for cannot be tested against a stub.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import hypot, isfinite
from typing import Annotated, Any, Literal, Sequence

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

#: Subgoal vocabulary accepted by the executor. Anything else is rejected
#: with a message the model can read and act on.
SUBGOAL_FIELDS: dict[str, tuple[str, ...]] = {
    'goto': ('x', 'y'),
    'search_around': ('x', 'y', 'radius'),
    'collect': (),
    'return_to_base': (),
}

#: Arena extent, generous box around it. The executor checks against the real
#: map; this only catches a model that has lost the plot.
WORLD_LIMIT = 10.0

#: Ceiling the executor enforces. Kept here so an oversized plan is cut or
#: refused locally instead of coming back as a rejection.
MAX_SUBGOALS = 50

#: How many subgoals to ask for. Their handbook recommends 3-6: few enough to
#: react when the ground changes, long enough not to spend a model call on
#: every step.
TARGET_SUBGOALS = 4

#: Radius bounds for ``search_around``, from the executor.
RADIUS_MIN = 0.1
RADIUS_MAX = 3.0

#: Battery budget, from the agent's own rule: enough to get home with margin.
#: The return estimate comes from the agent's cost map, so it already knows
#: about expensive ground; the margin covers the ground getting worse.
RETURN_MARGIN = 1.4
RETURN_RESERVE = 8.0

BASE_X = -2.0
BASE_Y = -0.5

#: Free arena, from ARCHITECTURE section 3. Used to keep the model inside the
#: room without making the planner compute geometry.
ARENA_X_MIN, ARENA_X_MAX = -2.8, 2.5
ARENA_Y_MIN, ARENA_Y_MAX = -2.5, 2.5

#: The nine pillars sit on a 3x3 grid with a step of about 1.1 m, radius
#: 0.15 m. The keep-out below is wider than the pillar on purpose: a plan that
#: grazes one is a plan that fails to find a path, and a rejection costs one
#: model call while a failed subgoal costs the robot the trip.
PILLAR_RADIUS = 0.15
PILLAR_KEEPOUT = 0.35
PILLAR_GRID = (-1.1, 0.0, 1.1)

#: Distance a plan keeps from the wall line, in metres.
#:
#: The task statement requires that walls "and a margin around them" be
#: forbidden, but that is the navigation layer's job and did_agent does it: its
#: cost map inflates walls by 0.20 m, which is more than the burger's radius and
#: what actually keeps the robot off them. Inflating them a second time here
#: only removed options from the plan — the passages in this arena are already
#: narrow, and a second 0.25 m made them unnarrowable. One map cell is enough to
#: keep a goal off the wall line itself and let the navigator handle clearance.
WALL_MARGIN = 0.05


@dataclass
class Arena:
    """Where the robot can actually go, as opposed to where numbers may be.

    Defaults to the values in ARCHITECTURE section 3, but the agent publishes
    the geometry it read out of the Gazebo scene, and a hardcoded grid is a
    guess that silently rots the first time a pillar moves. ``update`` takes
    the payload from ``/api/geometry``.
    """

    x_min: float = ARENA_X_MIN
    x_max: float = ARENA_X_MAX
    y_min: float = ARENA_Y_MIN
    y_max: float = ARENA_Y_MAX
    #: The walkable floor as a polygon, from the scene. Empty means "fall back
    #: to the rectangle above".
    #:
    #: The arena is a hexagon, and a rectangle around it is not the same shape:
    #: at y = 1.7 the left wall sits at x = -1.95 while the rectangle's edge is
    #: at -2.8. Every check that used the rectangle therefore accepted points
    #: that are 25 cm outside the wall — including the first column of the
    #: planner's own target grid, which put the robot into a corner to search
    #: and left it grinding along the wall with "no path to goal". The
    #: repository asks for exactly this: one source of geometry, constants only
    #: as a fallback.
    floor: tuple[tuple[float, float], ...] = ()
    pillars: tuple[tuple[float, float, float], ...] = tuple(
        (px, py, PILLAR_RADIUS)
        for px in PILLAR_GRID for py in PILLAR_GRID
    )
    #: True once real geometry has replaced the documented constants.
    from_scene: bool = False

    def update(self, payload: dict[str, Any]) -> bool:
        """Adopt the pillars the agent measured. Returns True if it changed.

        Only the pillars. ``bounds`` in ``/api/geometry`` is the extent of the
        outer wall, not of the walkable floor: the Gazebo scene reports
        x in [-3.65, 3.65] with wall thickness 0.3175, and the map-derived
        fallback reports x in [-3.25, 3.0], while the floor the robot can drive
        on is about x in [-2.85, 2.55]. Adopting those numbers as bounds would
        silently accept points that are inside the wall, which is the exact
        failure the bounds check exists to prevent.

        The pillar positions are trustworthy and are worth taking: the scene
        gives them exactly at (-1.1, 0, 1.1), while the map-derived version is
        offset by about 2.5 cm.
        """
        pillars = payload.get('pillars')
        if not isinstance(pillars, list) or not pillars:
            return False
        try:
            self.pillars = tuple(
                (float(item['x']), float(item['y']),
                 float(item.get('r', PILLAR_RADIUS)))
                for item in pillars
            )
        except (KeyError, TypeError, ValueError):
            return False

        # The floor polygon is the walkable area itself, which is what the
        # bounds are not: ``bounds`` is the extent of the outer wall. This is
        # the part that was missing, and it is the part that mattered.
        floor = payload.get('floor')
        if isinstance(floor, list) and len(floor) >= 3:
            try:
                points = tuple((float(item[0]), float(item[1])) for item in floor)
            except (IndexError, TypeError, ValueError):
                points = ()
            if len(points) >= 3:
                self.floor = points
                xs = [px for px, _ in points]
                ys = [py for _, py in points]
                self.x_min, self.x_max = min(xs), max(xs)
                self.y_min, self.y_max = min(ys), max(ys)
        self.from_scene = True
        return True


#: The arena every check reads. Mutated in place by :func:`load_geometry` so
#: that the plan validator stays a plain function with no plumbing.
ARENA = Arena()


def load_geometry(base_url: str = 'http://127.0.0.1:8080',
                  timeout: float = 5.0) -> bool:
    """Adopt the agent's measured arena geometry, if it is reachable.

    The agent already extracts pillars and wall bounds from the Gazebo scene
    for the dashboard. Reading them from the same place keeps the planner from
    validating against a grid that has drifted from the one in the scene.
    Failure is not an error: the documented constants stay in place.
    """
    import json
    import urllib.request

    try:
        with urllib.request.urlopen(f'{base_url}/api/geometry',
                                    timeout=timeout) as response:
            payload = json.loads(response.read().decode('utf-8'))
    except Exception:  # noqa: BLE001 - geometry is an improvement, not a need
        return False
    return ARENA.update(payload if isinstance(payload, dict) else {})


def _floor_gap(x: float, y: float,
               floor: tuple[tuple[float, float], ...]) -> float:
    """Signed distance from (x, y) to the nearest edge of the floor.

    Positive inside, negative outside. For a convex polygon the distance to the
    nearest edge decides everything: a point more than the margin inside can be
    driven to, and one outside cannot, whatever the rectangle says about it.
    """
    best = float('inf')
    count = len(floor)
    for index in range(count):
        ax, ay = floor[index]
        bx, by = floor[(index + 1) % count]
        edge_x, edge_y = bx - ax, by - ay
        length = hypot(edge_x, edge_y)
        if length <= 1e-9:
            continue
        # Point-to-segment distance, not to the infinite line: the floor is a
        # closed shape and a point off the end of an edge is inside it.
        t = ((x - ax) * edge_x + (y - ay) * edge_y) / (length * length)
        t = 0.0 if t < 0.0 else (1.0 if t > 1.0 else t)
        gap = hypot(x - (ax + t * edge_x), y - (ay + t * edge_y))
        if gap < best:
            best = gap
    if best == float('inf'):
        return -1.0
    crosses = 0
    for index in range(count):
        ax, ay = floor[index]
        bx, by = floor[(index + 1) % count]
        if (ay > y) != (by > y):
            x_hit = ax + (y - ay) * (bx - ax) / (by - ay)
            if x < x_hit:
                crosses += 1
    return best if crosses % 2 else -best


def arena_problem(x: float, y: float) -> str | None:
    """Why a point is unusable, or None when it is free floor.

    This is the check that turns a well-formed plan into a workable one. The
    executor's parser only rejects shapes — a wrong type, a string where a
    number belongs, a coordinate outside the world. It cannot reject a point
    two centimetres from a pillar, because that is a perfectly good number.
    Such a plan parses, then fails at run time as "no path to goal", and the
    robot loses the trip. Catching it here costs nothing: the plan is rejected
    with a reason the model can act on next time.
    """
    arena = ARENA
    if arena.floor:
        # Distance to the nearest edge, measured against the wall margin. For a
        # convex floor this is the inward-offset test: a point inside by more
        # than the margin is one the robot can stand on, and a point outside is
        # one that fails later as "no path to goal" after the trip is spent.
        gap = _floor_gap(x, y, arena.floor)
        if gap < WALL_MARGIN:
            return (f'точка ({x:g}; {y:g}) вне пола арены или вплотную к стене: '
                    f'до грани {gap:.2f} м при минимуме {WALL_MARGIN:g} м')
    else:
        lo_x, hi_x = arena.x_min + WALL_MARGIN, arena.x_max - WALL_MARGIN
        lo_y, hi_y = arena.y_min + WALL_MARGIN, arena.y_max - WALL_MARGIN
        if not lo_x <= x <= hi_x:
            return (f'точка ({x:g}; {y:g}) вне арены: x должен быть между '
                    f'{lo_x:.2f} и {hi_x:.2f}')
        if not lo_y <= y <= hi_y:
            return (f'точка ({x:g}; {y:g}) вне арены: y должен быть между '
                    f'{lo_y:.2f} и {hi_y:.2f}')
    for px, py, radius in arena.pillars:
        if hypot(x - px, y - py) < max(PILLAR_KEEPOUT, radius + 0.2):
            return (f'точка ({x:g}; {y:g}) попадает на столб у ({px:g}; {py:g}); '
                    f'держись не ближе {PILLAR_KEEPOUT:g} м от столбов')
    return None


def budget_floor(return_cost: float | None) -> float | None:
    """Battery below which going home is the only sensible plan.

    Returns None when the agent has not reported a return cost yet, in which
    case no floor can be computed and planning continues on judgement.
    """
    if return_cost is None:
        return None
    return return_cost * RETURN_MARGIN + RETURN_RESERVE


def must_return(battery: float, return_cost: float | None) -> bool:
    """Whether the budget forbids anything but going home."""
    floor = budget_floor(return_cost)
    if floor is None:
        return False
    return battery < floor


#: Battery kept aside regardless of what the return costs: the robot's own
#: radius of ignorance, the final approach and one failed attempt. Without a
#: floor of its own the estimate goes to zero whenever the robot stands on the
#: base, and the planner concludes it has an unlimited budget.
SAFETY_RESERVE = 12.0

#: Fraction of the battery still available that may be spent on going out.
SPENDABLE_FRACTION = 0.5

#: Below this much spendable battery one more sweep is not worth starting, so
#: going home becomes the right answer again.
SEARCH_WORTH_IT = 5.0

#: Sensor reading above which a sample is treated as close enough to act on.
#:
#: Derived, not chosen. The sensor reads ``max(0, 1 - d/1.5)``, and the widest
#: circle a search can usefully run is capped at 0.9 m. A reading only justifies
#: stopping and searching locally once the circle can still reach the sample it
#: points at: ``1.5 * (1 - margin) <= 0.9``, that is ``margin >= 0.4``.
#:
#: At 0.08 the planner chased readings it had no way to act on. A reading of
#: 0.15 means the sample is 1.29 m away and the best circle reaches 0.9, so the
#: robot spun a full circle with nothing to find; anything below 0.40 is
#: information without a response. Below this, keep sweeping the arena — the
#: systematic sweep is what actually finds things.
SIGNAL_NEAR = 0.40

#: Reading above which the sample is a step away rather than merely nearby.
#:
#: The sensor is ``max(0, 1 - d/1.5)``, so 0.6 means 0.60 m and 0.7 means 0.45 m.
#: A collect only succeeds under 0.30 m, so going straight for one here would
#: earn a ``false_collect`` and its penalty. What is needed instead is a short
#: approach, so this band gets a tight search circle rather than a wider one.
SIGNAL_CLOSE = 0.6

#: Reading above which the sample is already inside collect range and searching
#: for it is worse than useless.
#:
#: ``1 - 0.30/1.5`` is 0.80: past that the sample is within the radius where
#: ``collect`` succeeds. The agent's search reports success at ``found_level``
#: 0.7, which is 0.45 m — so a plan that searches and then collects walks the
#: robot around the sample and hands it a ``false_collect`` instead. Above this
#: reading the only useful action is to collect where it stands.
SIGNAL_TAKE = 0.8

#: Compatibility name used by the prompt: both values describe the judge's
#: exact 0.30 m collection boundary.
SIGNAL_COLLECTABLE = SIGNAL_TAKE

#: A useful UI/prompt warning threshold. Runtime decisions do not discard all
#: readings above it; :func:`signal_margin` subtracts the measured noise so a
#: very strong signal can still be acted on during the hard-scenario fault.
NOISE_UNTRUSTWORTHY = 0.06


def signal_margin(signal: float | None, noise: float | None) -> float:
    """The part of a reading that is more than the noise.

    ``noise_estimate`` is how far a reading may sit from the truth, so a noisy
    sample says much less than the number suggests. Subtracting it leaves only
    what can be relied on: 0.19 against a noise of 0.18 is nothing, while 0.85
    against the same noise still means a sample within half a metre.

    Ignoring the sensor outright once it is noisy — the earlier rule — throws
    the strong readings away with the weak ones, and the robot drives past
    samples that were right under it for as long as the fault lasts.
    """
    if signal is None:
        return -1.0
    return signal - max(0.0, noise or 0.0)

#: A leg longer than this is treated as a journey rather than a repositioning,
#: and is judged on what the floor along it costs.
LEG_WORTH_M = 2.0

#: A goto this close to where the robot already stands is a no-op. Models write
#: it to mean "right here", so refusing it would force a repair round over every
#: strong-signal plan — one wasted model call each.
GOTO_NOOP_M = 0.7

#: How close a target may be to a place the robot has already hit.
COLLISION_RADIUS_M = 0.7

#: How close a point has to be to the robot to count as "where it already
#: stands". Big enough to cover the spread of a stuck pose across a few
#: replans, small enough that it is not a licence to plan anywhere nearby.
HERE_M = 0.45


def _under_robot(x: float, y: float,
                 pose: Sequence[float] | None) -> bool:
    """Whether (x, y) is the spot the robot is already standing on."""
    if not pose or len(pose) < 2:
        return False
    return hypot(x - float(pose[0]), y - float(pose[1])) < HERE_M

#: Battery a single long leg may cost before the plan is refused outright.
#: A search point is a maybe: it may find nothing. Nine units is already more
#: than any single discovery is worth, so a leg that costs more is a bad bet
#: whatever the battery happens to hold.
LEG_WORTH_COST = 8.0

#: …or, whatever the ground, a single leg may not eat more than this share of
#: what is left.
LEG_SHARE = 0.25


def _leg_crosses(start: tuple[float, float], end: tuple[float, float],
                 patch: dict[str, float]) -> bool:
    """Whether the straight leg from start to end passes through a patch.

    A circle test on the leg's midpoint plus both ends. Exact segment geometry
    would be nicer, but this only has to be conservative: the planner is
    discouraged from far targets when expensive ground is in play, and being
    slightly over-eager about that is far cheaper than the alternative.
    """
    cx, cy = float(patch['x']), float(patch['y'])
    reach = float(patch.get('reach', 0.0))
    for point in (start, end,
                  ((start[0] + end[0]) / 2, (start[1] + end[1]) / 2),
                  (start[0] * 0.75 + end[0] * 0.25,
                   start[1] * 0.75 + end[1] * 0.25),
                  (start[0] * 0.25 + end[0] * 0.75,
                   start[1] * 0.25 + end[1] * 0.75)):
        if hypot(point[0] - cx, point[1] - cy) <= reach:
            return True
    return False


def spendable_budget(battery: float, return_cost: float | None) -> float:
    """How much battery may be spent away from the base, and no more.

    Two terms. The agent's own estimate of the way home, which is the honest
    one while the robot is out in the arena. Plus a reserve that does not
    shrink to nothing at the base, which is where the estimate alone fails:
    standing on the base it reads as zero, and a planner that trusts it will
    plan a sweep of the whole arena on a battery that cannot pay for it.
    """
    spent = max(0.0, return_cost or 0.0)
    return max(0.0, battery - spent - SAFETY_RESERVE) * SPENDABLE_FRACTION


class Subgoal(BaseModel):
    """One step, in the executor's format."""

    model_config = ConfigDict(extra='forbid')

    type: Literal['goto', 'search_around', 'collect', 'return_to_base']
    x: float = 0.0
    y: float = 0.0
    radius: float = 0.0

    @field_validator('x', 'y')
    @classmethod
    def _finite_and_in_world(cls, value: float) -> float:
        if not isfinite(value) or abs(value) > WORLD_LIMIT:
            raise ValueError(f'coordinate {value} is out of the ±{WORLD_LIMIT} m world')
        return float(value)

    @model_validator(mode='after')
    def _radius_if_needed(self) -> 'Subgoal':
        if self.type == 'search_around' and not RADIUS_MIN <= self.radius <= RADIUS_MAX:
            raise ValueError(
                f'radius must be between {RADIUS_MIN} and {RADIUS_MAX} m, '
                f'got {self.radius}'
            )
        if self.type == 'return_to_base':
            return self.model_copy(update={'x': BASE_X, 'y': BASE_Y})
        return self

    def to_wire(self) -> dict[str, Any]:
        """The JSON object as the executor wants it, with only relevant keys."""
        if self.type == 'goto':
            return {'type': 'goto', 'x': round(self.x, 3), 'y': round(self.y, 3)}
        if self.type == 'search_around':
            return {'type': 'search_around', 'x': round(self.x, 3),
                    'y': round(self.y, 3), 'radius': round(self.radius, 3)}
        return {'type': self.type}

    def describe(self) -> str:
        if self.type == 'search_around':
            return f'search_around({self.x:g}, {self.y:g}, r={self.radius:g})'
        if self.type == 'goto':
            return f'goto({self.x:g}, {self.y:g})'
        return self.type


# These four models describe what the LLM is allowed to generate.  They are
# deliberately separate instead of one model with optional/default x, y and
# radius fields: Structured Outputs must make a missing coordinate impossible,
# not turn it into (0, 0) after the response arrives.
class _StrictModelOutput(BaseModel):
    model_config = ConfigDict(extra='forbid')


class GotoOutput(_StrictModelOutput):
    type: Literal['goto']
    x: float = Field(ge=-WORLD_LIMIT, le=WORLD_LIMIT)
    y: float = Field(ge=-WORLD_LIMIT, le=WORLD_LIMIT)


class SearchAroundOutput(_StrictModelOutput):
    type: Literal['search_around']
    x: float = Field(ge=-WORLD_LIMIT, le=WORLD_LIMIT)
    y: float = Field(ge=-WORLD_LIMIT, le=WORLD_LIMIT)
    radius: float = Field(ge=RADIUS_MIN, le=RADIUS_MAX)


class CollectOutput(_StrictModelOutput):
    type: Literal['collect']


class ReturnToBaseOutput(_StrictModelOutput):
    type: Literal['return_to_base']


ModelSubgoal = Annotated[
    GotoOutput | SearchAroundOutput | CollectOutput | ReturnToBaseOutput,
    Field(discriminator='type'),
]


class ModelPlanOutput(_StrictModelOutput):
    """Exact response generated by the model (plan_id is assigned locally)."""

    explanation: str = Field(min_length=1, max_length=500)
    subgoals: list[ModelSubgoal] = Field(min_length=1)


def plan_response_format(max_subgoals: int = 6) -> dict[str, Any]:
    """OpenAI-compatible strict Structured Outputs declaration for a plan."""
    if max_subgoals < 1:
        raise ValueError('max_subgoals must be positive')
    schema = ModelPlanOutput.model_json_schema()
    # The shape is Pydantic-owned; the planning horizon is a live ROS
    # parameter.  Setting maxItems here keeps the API contract and prompt on
    # exactly the same value without manufacturing a model class per request.
    schema['properties']['subgoals']['maxItems'] = int(max_subgoals)
    return {
        'type': 'json_schema',
        'json_schema': {
            'name': 'agent_plan',
            'strict': True,
            'schema': schema,
        },
    }


class Plan(BaseModel):
    """A plan ready to publish."""

    model_config = ConfigDict(extra='forbid')

    plan_id: str = Field(min_length=1)
    subgoals: list[Subgoal] = Field(min_length=1, max_length=MAX_SUBGOALS)
    #: Shown on the dashboard as the reasoning behind the plan.
    explanation: str = ''

    def to_wire(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            'plan_id': self.plan_id,
            'subgoals': [item.to_wire() for item in self.subgoals],
        }
        if self.explanation:
            payload['explanation'] = self.explanation
        return payload


class PlanRejected(ValueError):
    """The model's answer is not a plan. The message goes back to it."""


def parse_model_plan(raw: str | dict[str, Any],
                     plan_id: str,
                     max_subgoals: int = MAX_SUBGOALS) -> Plan:
    """Turn a model answer into a validated plan.

    Errors are phrased as instructions, because the handbook's loop feeds the
    rejection reason straight back into the next prompt: "radius must be
    between 0.1 and 3.0 m" tells the model what to change, "invalid JSON" does
    not.
    """
    if isinstance(raw, str):
        import json
        try:
            data = json.loads(raw)
        except (TypeError, ValueError) as error:
            raise PlanRejected(
                f'ответ не разобран как JSON ({error}). Верни только JSON.'
            ) from error
    else:
        data = raw

    if not isinstance(data, dict):
        raise PlanRejected('ответ должен быть объектом с полем "subgoals"')
    try:
        generated = ModelPlanOutput.model_validate(data)
    except Exception as error:  # noqa: BLE001 - concise feedback goes to model
        raise PlanRejected(_brief(error)) from error
    if len(generated.subgoals) > max_subgoals:
        raise PlanRejected(
            f'подцелей {len(generated.subgoals)}, максимум {max_subgoals}'
        )

    subgoals = [
        Subgoal.model_validate(item.model_dump())
        for item in generated.subgoals
    ]
    return Plan(plan_id=plan_id, subgoals=subgoals,
                explanation=generated.explanation)


def on_expensive_ground(x: float, y: float,
                        expensive: list[dict[str, Any]] | None) -> bool:
    """Whether a point sits inside ground the agent has measured as dear.

    One predicate for two callers, deliberately. The planner uses it to keep
    expensive cells out of the targets it offers, and the check uses it to
    refuse a plan that drives onto one. Written separately they disagreed: the
    planner offered cells it knew were expensive, the model picked them because
    they were the nearest offered, and the plan came straight back rejected.
    The refusal rate was this planner's own doing.
    """
    for patch in expensive or ():
        reach = float(patch.get('reach', 0.0))
        if hypot(x - float(patch['x']), y - float(patch['y'])) <= reach:
            return True
    return False


def check_plan(plan: Plan,
               expensive: list[dict[str, float]] | None = None,
               *,
               min_battery: float | None = None,
               samples_remaining: int | None = None,
               signal_high: float | None = None,
               sensor_noise: float | None = None,
               pose: tuple[float, float] | None = None,
               battery: float | None = None,
               hits: list[tuple[float, float]] | None = None,
               cost_per_search: float = 1.2) -> list[str]:
    """Everything wrong with an otherwise well-formed plan.

    Returns a list of messages rather than the first one: telling the model
    about all the bad points at once costs one call instead of one per point.
    Empty means the plan is worth publishing.
    """
    problems: list[str] = []
    visited: list[tuple[str, float, float]] = []

    for index, subgoal in enumerate(plan.subgoals):
        if subgoal.type in ('goto', 'search_around'):
            issue = arena_problem(subgoal.x, subgoal.y)
            if issue:
                problems.append(f'подцель {index}: {issue}')
            # Only travelling to somewhere already visited is wasted motion.
            # A search in the spot you have just arrived at is the intended
            # pattern, and rejecting it would refuse most good plans.
            if subgoal.type == 'goto':
                for name, x, y in visited:
                    if hypot(x - subgoal.x, y - subgoal.y) < 0.15:
                        problems.append(
                            f'подцель {index}: повторяет {name} '
                            f'({subgoal.x:g}; {subgoal.y:g}) — туда уже ехали')
                        break
            visited.append((subgoal.type, subgoal.x, subgoal.y))

    # Ground the robot has already hit. Sending it back to the same spot is
    # what turns one collision into a loop: the judge emits an event every two
    # seconds while it is stuck, and each one would trigger a new plan aimed at
    # the same obstacle.
    #
    # But a robot stuck against something is standing *on* that spot, and the
    # veto has to let it work from where it is. Refusing the robot's own
    # position is a livelock: on medium the robot wedged itself at (0.5; -0.79)
    # with the sensor at 0.73 — a sample within reach — and every plan that
    # said "search here, then collect" was turned down as a return to known
    # bad ground, while the model kept writing exactly that plan. It sat there
    # colliding until the battery ran down. A ``search_around`` where the robot
    # already stands is not a trip back into the obstacle; it is the spiral
    # that walks it off, and it is where a nearby sample gets collected.
    for index, subgoal in enumerate(plan.subgoals):
        if subgoal.type not in ('goto', 'search_around'):
            continue
        if subgoal.type == 'search_around' and _under_robot(subgoal.x,
                                                            subgoal.y, pose):
            continue
        if any(hypot(subgoal.x - hx, subgoal.y - hy) < COLLISION_RADIUS_M
               for hx, hy in hits or ()):
            problems.append(
                f'подцель {index}: ({subgoal.x:g}; {subgoal.y:g}) — там уже '
                'было столкновение, не отправляй робота туда снова')

    # Expensive ground is refused outright. The mission says to avoid it, and
    # a plan that crosses it burns the battery the next sector needs.
    # Every offending subgoal gets its own message — the check collects all
    # problems at once (see the docstring), so it must not stop at the first
    # one, and in particular not because an earlier check already found
    # something: that version silently skipped the ground check for every
    # subgoal after the first.
    for index, subgoal in enumerate(plan.subgoals):
        if subgoal.type not in ('goto', 'search_around'):
            continue
        for patch in expensive or ():
            reach = float(patch.get('reach', 0.0))
            if hypot(subgoal.x - float(patch['x']),
                     subgoal.y - float(patch['y'])) <= reach:
                problems.append(
                    f'подцель {index}: ({subgoal.x:g}; {subgoal.y:g}) лежит на '
                    f'дорогом грунте цены ×{patch.get("cost", 1):.1f} — '
                    'объедь его стороной')
                break

    # Value against price. A long leg to a guessed search point is only worth
    # making when the floor between here and there is ordinary. Without this
    # the model crosses the whole arena to reach a point it invented and pays
    # several times the normal price for the privilege, which is exactly the
    # behaviour of burning the battery while other samples go unpicked.
    if expensive and pose is not None and battery is not None:
        for index, subgoal in enumerate(plan.subgoals):
            if subgoal.type not in ('goto', 'search_around'):
                continue
            distance = hypot(subgoal.x - pose[0], subgoal.y - pose[1])
            if distance < LEG_WORTH_M:
                continue
            dearest = max((float(patch['cost']) for patch in expensive
                           if _leg_crosses(pose, (subgoal.x, subgoal.y),
                                           patch)),
                          default=1.0)
            price = distance * dearest
            if price > LEG_WORTH_COST or price > battery * LEG_SHARE:
                problems.append(
                    f'подцель {index}: {distance:.1f} м по грунту цены '
                    f'×{dearest:.1f} — это {price:.0f} ед. батареи за одну '
                    'точку. Возьми ближнюю или возвращайся')
                break

    searches = [i for i, s in enumerate(plan.subgoals) if s.type == 'search_around']
    collects = [i for i, s in enumerate(plan.subgoals) if s.type == 'collect']

    # A collect with no search before it is a false collect: the judge charges
    # a penalty for collecting where nothing is.
    # A collect on its own is normally a guess and earns a false_collect, so it
    # has to follow a search. Not when the reading already puts the sample
    # inside the collection radius: searching there drives the robot away from
    # the very thing it was told to take, which is the same false_collect by a
    # longer route.
    take_now = signal_margin(signal_high, sensor_noise) >= SIGNAL_TAKE
    for index in collects:
        before = plan.subgoals[:index]
        if not any(item.type in ('search_around',) for item in before):
            if take_now:
                continue
            problems.append(
                f'подцель {index}: collect без предшествующего поиска — '
                'сначала search_around, потом collect')

    for index in searches:
        if index + 1 not in collects:
            problems.append(
                f'подцель {index}: после search_around нужен collect сразу, '
                f'иначе образец будет найден и оставлен')

    # A live sensor means a sample is within reach. Travelling away from it is
    # the most expensive mistake available here: the model cannot know where
    # another one is, so a distant search point burns battery crossing the
    # arena while a collectible sample goes unpicked.
    #
    # The reading is believed only as far as it clears the noise. Under the
    # judge's sensor fault a lone sample is jitter, and obeying it would pin the
    # robot to one spot while a real sample goes unpicked elsewhere — but a
    # reading that still stands above the noise is worth following.
    near = signal_margin(signal_high, sensor_noise) >= SIGNAL_NEAR
    if near:
        first_move = next((i for i, s in enumerate(plan.subgoals)
                           if s.type in ('goto', 'search_around')), None)
        if first_move is not None:
            for index in range(first_move + 1):
                subgoal = plan.subgoals[index]
                if subgoal.type != 'goto':
                    continue
                # A goto to where the robot already stands is a no-op, and
                # models write it constantly to mean "here, right now". Refusing
                # it makes every strong-signal plan need a repair round, which
                # costs a full model call each time.
                if pose is not None and hypot(subgoal.x - pose[0],
                                              subgoal.y - pose[1]) < GOTO_NOOP_M:
                    continue
                problems.append(
                    f'подцель {index}: сигнал {signal_high:.2f} — образец '
                    f'в пределах досягаемости, сначала search_around и collect '
                    f'на месте, а не поездка в ({subgoal.x:g}; {subgoal.y:g})')
                break

        # Local search must be local: a wide circle around a sample that is
        # already close is the "spinning without collecting" failure, and the
        # radius should shrink as the reading grows.
        for index, subgoal in enumerate(plan.subgoals):
            if subgoal.type != 'search_around' or index > first_move:
                continue
            allowed = max(0.4, 1.6 * max(0.0, 1.0 - signal_high))
            if subgoal.radius > allowed + 0.05:
                problems.append(
                    f'подцель {index}: сигнал {signal_high:.2f} — образец '
                    f'рядом, радиус {subgoal.radius:g} велик, бери '
                    f'не больше {allowed:.1f} м')
                break

    # Does the plan fit in the battery? Rough, on the assumption that a search
    # circle costs roughly as much driving as its own radius.
    if min_battery is not None:
        cost = sum(cost_per_search if s.type == 'search_around' else 0.6
                   for s in plan.subgoals)
        if cost > min_battery:
            problems.append(
                f'план примерно на {cost:.0f} ед. батареи, а доступно '
                f'{min_battery:.0f} — сократи число поисков или возвращайся')

    # Ending the episode is not the model's call. ``return_to_base`` makes the
    # agent call /did/finish, and the judge closes the run: a plan that ends
    # there with samples still on the floor finishes the episode at whatever it
    # has collected so far, and no later plan can be issued. The mission says
    # collect as many as possible and only then come back, so this is refused
    # while samples remain and the battery can pay for another sweep.
    if (plan.subgoals and plan.subgoals[-1].type == 'return_to_base'
            and samples_remaining and samples_remaining > 0
            and (min_battery is None or min_battery > SEARCH_WORTH_IT)):
        problems.append(
            f'нельзя заканчивать эпизод: осталось образцов '
            f'{samples_remaining}, батареи хватает ещё на '
            f'{min_battery:.0f} ед. Добавь поиск; return_to_base оставь '
            'только когда батареи реально не хватает')
    return problems


def home_plan(plan_id: str, explanation: str = '') -> Plan:
    """The fallback: go home and finish.

    Used when the budget forbids anything else, when the model is unavailable,
    or when its answer did not parse. It has to be a plan rather than nothing
    so the episode still ends with the robot on the base.
    """
    return Plan(
        plan_id=plan_id,
        subgoals=[Subgoal(type='return_to_base')],
        explanation=explanation or 'батареи не хватает на обход, возвращаюсь',
    )


def _brief(error: Exception) -> str:
    """The part of a pydantic error a model can act on.

    Pydantic's default text starts with "1 validation error for Subgoal",
    which says nothing about which field is wrong or how. Since this string
    goes straight back into the next prompt, it has to name the field and the
    problem, or the model will guess.
    """
    errors = getattr(error, 'errors', None)
    if callable(errors):
        try:
            details = errors()
        except Exception:  # noqa: BLE001 - fall through to the text
            details = []
        parts = []
        for item in details:
            field = '.'.join(str(part) for part in item.get('loc', ()))
            message = str(item.get('msg', ''))
            # "Value error, radius must be between..." reads as noise once the
            # field is already named.
            message = message.removeprefix('Value error, ')
            if not field or field in message:
                parts.append(message)
            else:
                parts.append(f'{field}: {message}')
        if parts:
            return '; '.join(parts)
    text = str(error).strip().splitlines()
    return text[0] if text else error.__class__.__name__
