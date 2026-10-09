"""A Robot backed by the kinematic simulator and the judge model (no ROS, no Gazebo).

Used by the tests and by ``python -m did_agent.bench`` to run whole episodes in
seconds. It follows the same rules as the real stack: the pose seen by the judge
is the start pose plus odometry, collisions come from the lidar, and the agent
only learns what the public topics tell it.
"""

from __future__ import annotations

from collections.abc import Callable
from statistics import median

from did_judge.judge_model import JudgeModel
from did_judge.judge_model import scan_clearance
from did_judge.scenario import Scenario

from did_agent.adaptation import Adaptation
from did_agent.controller import Pose
from did_agent.costmap import CostMap
from did_agent.grid import GridMap
from did_agent.grid import load_map
from did_agent.kinematic_sim import KinematicSim
from did_agent.navigator_core import DONE
from did_agent.navigator_core import FAILED
from did_agent.navigator_core import NavigatorCore
from did_agent.robot import NavResult
from did_agent.robot import Reading
from did_agent.robot import sample_signal_near

TICK = 0.05


class SimRobot:
    """Implements did_agent.robot.Robot on top of the kinematic simulation."""

    def __init__(
        self,
        scenario: Scenario,
        grid: GridMap | None = None,
        *,
        hidden: list[tuple[float, float, float]] | None = None,
        costmap: CostMap | None = None,
        yaw_noise: float = 0.0,
        learn: bool = True,
        on_tick: Callable[['SimRobot'], None] | None = None,
    ) -> None:
        self.grid = grid or load_map()
        self.scenario = scenario
        self.costmap = costmap or CostMap(self.grid)
        self.sim = KinematicSim(
            self.grid,
            Pose(scenario.base_x, scenario.base_y, 0.0),
            hidden=hidden,
            yaw_noise=yaw_noise,
        )
        self.judge = JudgeModel(scenario)
        self.nav = NavigatorCore(self.costmap)
        self.on_tick = on_tick
        self.preempt = False
        self.events: list[dict] = []
        self.journal_log: list[dict] = []
        self.adaptation = Adaptation(
            self.costmap,
            lambda kind, title, text, status: self.journal_log.append(
                {'t': round(self.sim.t, 1), 'kind': kind, 'title': title,
                 'text': text, 'status': status}),
            learn=learn,
        )
        self._ticks = 0
        self._seen_events = 0
        self._sensor_value = 0.0
        self._sensor_count = 0
        self._previous_battery = self.judge.battery

    # --- time ---------------------------------------------------------------------

    def _tick(self, linear: float, angular: float) -> None:
        self.sim.step(linear, angular, TICK)
        self.judge.advance(self.sim.t)
        self.events += self.judge.update_odometry(
            self.sim.pose.x - self.scenario.base_x,
            self.sim.pose.y - self.scenario.base_y,
        )
        scan = self.sim.scan()
        clearance = scan_clearance(scan.angles(), scan.ranges)
        event = self.judge.report_scan(clearance) if clearance is not None else None
        if event:
            self.events.append(event)
        self._feed_adaptation(event)
        if self.on_tick:
            self.on_tick(self)

    def _feed_adaptation(self, collision: dict | None) -> None:
        """Give the adaptation what the real topics would: battery, pose, events at 10 Hz."""
        self._ticks += 1
        for event in self.events[self._seen_events:]:
            self.adaptation.on_event(self.sim.t, event)
        self._seen_events = len(self.events)
        if self._ticks % 2:
            return
        # The battery topic lags the odometry a little, as in the real stack.
        self.adaptation.on_battery(self.sim.t, self._previous_battery)
        self._previous_battery = self.judge.battery
        self.adaptation.on_pose(self.sim.t, self.sim.pose.x, self.sim.pose.y)
        # The sample sensor streams at 10 Hz all the time, not only when we stand still.
        self._sensor_value = self.judge.sample_sensor()
        self._sensor_count += 1
        self.adaptation.on_sensor(self.sim.t, self._sensor_value)

    # --- Robot interface --------------------------------------------------------------

    def pose(self) -> Pose:
        return Pose(self.sim.pose.x, self.sim.pose.y, self.sim.pose.yaw)

    def now(self) -> float:
        return self.sim.t

    def preempted(self) -> bool:
        return self.preempt

    def goto(
        self,
        x: float,
        y: float,
        timeout: float = 180.0,
        guard: Callable[[], bool] | None = None,
        stop_on_signal: bool = False,
    ) -> NavResult:
        if not self.nav.set_goal((x, y), self.pose(), self.sim.t):
            return NavResult(FAILED, self.nav.reason)
        start, next_guard = self.sim.t, self.sim.t + 1.0
        while self.sim.t - start < timeout and not self.preempt:
            command = self.nav.update(self.pose(), self.sim.scan(), self.sim.t)
            self._tick(command.linear, command.angular)
            if self.nav.status in (DONE, FAILED):
                break
            if (guard is not None or stop_on_signal) and self.sim.t >= next_guard:
                next_guard = self.sim.t + 1.0
                if guard is not None and guard():
                    self.nav.cancel()
                    return NavResult(FAILED, 'battery reserve reached')
                if stop_on_signal and sample_signal_near(
                    self._sensor_value, self.adaptation.monitor.noise_estimate,
                ):
                    self.nav.cancel()
                    return NavResult(FAILED, 'sample signal nearby')
        else:
            self.nav.cancel()
            return NavResult(FAILED, 'preempted' if self.preempt else 'timeout')
        return NavResult(
            self.nav.status,
            self.nav.reason,
            self.nav.distance_to_goal(self.pose()),
            self.nav.replans,
        )

    def read_sensor(self, count: int = 5) -> Reading:
        values = []
        last = self._sensor_count
        while len(values) < count:
            self._tick(0.0, 0.0)
            if self._sensor_count != last:
                last = self._sensor_count
                values.append(self._sensor_value)
        mid = median(values)
        spread = 1.4826 * median(abs(v - mid) for v in values)
        return Reading(mid, spread)

    def noise_level(self) -> float:
        return self.adaptation.noise_level()

    def anomaly(self) -> dict[str, bool]:
        return self.adaptation.monitor.anomaly(self.sim.t)

    def collect(self) -> tuple[bool, str]:
        ok, event = self.judge.collect()
        self.events.append(event)
        return ok, 'sample collected' if ok else 'no sample within radius'

    def finish(self) -> tuple[bool, str]:
        ok = self.judge.finish()
        return ok, 'run finished at base' if ok else 'robot is not at base'

    def battery(self) -> float:
        return self.judge.battery

    def samples_total(self) -> int:
        return len(self.judge.samples)

    def collected(self) -> int:
        return self.judge.collected_count
