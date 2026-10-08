"""Entry point: the LLM planner as a ROS node.

    ros2 run did_llm llm_planner
    ros2 run did_llm llm_planner --ros-args -p mission:='собери образцы'

The node publishes plans and nothing else. It never touches /cmd_vel, never
publishes to a judge topic and never reads the judge's internals: its inputs
are /agent/state and /agent/status, its outputs are /agent/plan and, only when
the model is unreachable, /agent/command.
"""

from __future__ import annotations

import os

import rclpy
from rclpy.node import Node

from did_llm.agent_link import AgentLink
from did_llm.journal import Journal, JournalConfig
from did_llm.agent_plan import ARENA, load_geometry
from did_llm.llm_client import LLMClient, LLMConfig, load_api_key
from did_llm.planner_node import Planner, PlannerConfig

DEFAULT_MISSION = 'Собрать образцы, избегать дорогих зон, вернуться на базу'


class LLMPlanner(Node):
    """Ties the link, the client and the loop together."""

    def __init__(self) -> None:
        super().__init__('llm_planner')

        self.declare_parameter('mission', DEFAULT_MISSION)
        self.declare_parameter('base_url', 'https://api-ai.mai.ru/v1')
        self.declare_parameter('model', 'DeepSeek-V4-Flash')
        self.declare_parameter('reasoning_effort', 'none')
        self.declare_parameter('max_calls_per_minute', 20)
        self.declare_parameter('max_calls_total', 400)
        self.declare_parameter('min_interval_sec', 4.0)
        self.declare_parameter('timeout_sec', 180.0)
        self.declare_parameter('replan_period_sec', 30.0)
        self.declare_parameter('min_subgoals', 3)
        self.declare_parameter('max_subgoals', 6)
        self.declare_parameter('repair_attempts', 1)
        self.declare_parameter('autonomous_fallback', True)
        self.declare_parameter('exchanges_path', 'logs/did_session.jsonl')
        self.declare_parameter('tick_period_sec', 1.0)

        def param(name, caster=None):
            value = self.get_parameter(name).value
            return caster(value) if caster else value

        self.link = AgentLink(self)
        self.link.log = self.get_logger()
        # Adopt the arena the agent measured, so plans are validated against
        # the geometry in the scene rather than against documented constants.
        # The dashboard answers late: it starts with the planner and only
        # settles once Gazebo has answered its own query, so the fetch is
        # retried until it works rather than tried once and forgotten.
        self.declare_parameter('agent_api_url', 'http://127.0.0.1:8080')
        self._api_url = param('agent_api_url')
        self._geometry_tries = 0
        self._geometry_timer = None
        if not self._refresh_geometry():
            self.get_logger().info(
                'геометрия арены: значения по описанию, /api/geometry '
                'пока недоступен, будет повтор')
            self._geometry_timer = self.create_timer(10.0, self._retry_geometry)
        self.journal = Journal(
            JournalConfig(exchanges_path=param('exchanges_path')),
            logger=self.get_logger(),
        )
        self.client = LLMClient(
            LLMConfig(
                base_url=param('base_url'),
                api_key=load_api_key(),
                model=param('model'),
                reasoning_effort=param('reasoning_effort'),
                max_calls_per_minute=param('max_calls_per_minute', int),
                max_calls_total=param('max_calls_total', int),
                min_interval_sec=param('min_interval_sec', float),
                timeout_sec=param('timeout_sec', float),
            ),
            journal=self.journal,
            logger=self.get_logger(),
        )
        self.planner = Planner(
            self.link,
            self.client,
            PlannerConfig(
                mission=param('mission'),
                min_subgoals=param('min_subgoals', int),
                max_subgoals=param('max_subgoals', int),
                replan_period_sec=param('replan_period_sec', float),
                repair_attempts=param('repair_attempts', int),
                autonomous_fallback=param('autonomous_fallback', bool),
            ),
        )

        self.create_timer(param('tick_period_sec', float), self._tick)
        self.get_logger().info(
            f'LLM-планировщик готов: model={self.client.cfg.model}, '
            f'key={"set" if self.client.cfg.configured else "missing"}'
        )

    def _tick(self) -> None:
        try:
            self.planner.tick()
        except Exception as error:  # noqa: BLE001 - the loop must not die
            self.get_logger().error(f'planning tick failed: {error}')

    def _refresh_geometry(self) -> bool:
        if load_geometry(self._api_url):
            self.get_logger().info('геометрия арены взята из сцены Gazebo')
            return True
        return False

    def _retry_geometry(self) -> None:
        """Keep asking until the agent's geometry shows up, then stop.

        After a few minutes the documented constants are as good as anything
        available, and the fallback message has already been said.
        """
        self._geometry_tries += 1
        if self._refresh_geometry() or self._geometry_tries >= 12:
            if not ARENA.from_scene:
                self.get_logger().info(
                    'геометрия арены остаётся по описанию: /api/geometry '
                    'не ответил, проверка использует константы')
            if self._geometry_timer is not None:
                self._geometry_timer.cancel()
                self._geometry_timer = None


def main(args=None) -> None:
    rclpy.init(args=args)
    node = LLMPlanner()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()