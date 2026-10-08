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
from did_llm.llm_client import LLMClient, LLMConfig
from did_llm.planner_node import Planner, PlannerConfig

DEFAULT_MISSION = 'Собрать образцы, избегать дорогих зон, вернуться на базу'


def _api_key() -> str:
    """The key is never logged and never published."""
    for name in ('DID_LLM_API_KEY', 'LLM_API_KEY', 'OPENAI_API_KEY'):
        value = os.environ.get(name)
        if value:
            return value
    for path in (os.path.join(os.getcwd(), '.env'),
                 os.path.join(os.path.expanduser('~'), '.env')):
        if not os.path.exists(path):
            continue
        with open(path, encoding='utf-8') as handle:
            for line in handle:
                line = line.strip()
                for key in ('DID_LLM_API_KEY', 'llm_api_key', 'OPENAI_API_KEY'):
                    if line.startswith(f'{key}='):
                        return line.split('=', 1)[1].strip().strip('"\'')
    return ''


class LLMPlanner(Node):
    """Ties the link, the client and the loop together."""

    def __init__(self) -> None:
        super().__init__('llm_planner')

        self.declare_parameter('mission', DEFAULT_MISSION)
        self.declare_parameter('base_url', 'https://api-ai.mai.ru/v1')
        self.declare_parameter('model', 'DeepSeek-V4-Flash')
        self.declare_parameter('max_calls_per_minute', 20)
        self.declare_parameter('max_calls_total', 400)
        self.declare_parameter('replan_period_sec', 12.0)
        self.declare_parameter('min_subgoals', 3)
        self.declare_parameter('max_subgoals', 6)
        self.declare_parameter('repair_attempts', 1)
        self.declare_parameter('exchanges_path', 'logs/did_session.jsonl')
        self.declare_parameter('tick_period_sec', 1.0)

        def param(name, caster=None):
            value = self.get_parameter(name).value
            return caster(value) if caster else value

        self.link = AgentLink(self)
        self.link.log = self.get_logger()
        self.journal = Journal(
            JournalConfig(exchanges_path=param('exchanges_path')),
            logger=self.get_logger(),
        )
        self.client = LLMClient(
            LLMConfig(
                base_url=param('base_url'),
                api_key=_api_key(),
                model=param('model'),
                max_calls_per_minute=param('max_calls_per_minute', int),
                max_calls_total=param('max_calls_total', int),
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