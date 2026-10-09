"""Entry point: the LLM planner as a ROS node.

    ros2 run did_llm llm_planner

The node publishes plans and nothing else. It never touches /cmd_vel, never
publishes to a judge topic and never reads the judge's internals: its inputs
are /agent/state, /agent/status and operator commands. It publishes a verified
goal selection on /agent/plan and a source-labelled explanation to the journal.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import rclpy
from rclpy.node import Node

from did_llm.agent_link import AgentLink
from did_llm.journal import Journal, JournalConfig
from did_llm.llm_client import LLMClient, LLMConfig, load_api_key
from did_llm.math_planner import MathPlanner


class LLMPlanner(Node):
    """Ties the link, the client and the loop together."""

    def __init__(self) -> None:
        super().__init__('llm_planner')

        self.declare_parameter('planner_mode', 'math_goals')
        self.declare_parameter('selection_policy', 'llm')
        self.declare_parameter('base_url', 'https://api-ai.mai.ru/v1')
        self.declare_parameter('model', 'deepseek-v4.1-flash')
        self.declare_parameter('reasoning_effort', 'none')
        self.declare_parameter('max_calls_per_minute', 20)
        self.declare_parameter('max_calls_total', 400)
        self.declare_parameter('min_interval_sec', 4.0)
        self.declare_parameter('timeout_sec', 30.0)
        self.declare_parameter('max_retries', 0)
        self.declare_parameter('exchanges_path', 'logs/did_session.jsonl')
        self.declare_parameter('metrics_path', '')
        self.declare_parameter('tick_period_sec', 1.0)

        def param(name, caster=None):
            value = self.get_parameter(name).value
            return caster(value) if caster else value

        mode = param('planner_mode')
        if mode != 'math_goals':
            raise ValueError('the live backend requires planner_mode=math_goals')
        selection_policy = param('selection_policy')
        if selection_policy not in ('llm', 'budget'):
            raise ValueError('selection_policy must be llm or budget')
        self.link = AgentLink(self, observed_only=True)
        self.link.log = self.get_logger()
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
                max_retries=param('max_retries', int),
            ),
            journal=self.journal,
            logger=self.get_logger(),
        ) if selection_policy == 'llm' else None
        self.planner = MathPlanner(self.link, self.client, selection_policy=selection_policy)
        self._metrics_path = param('metrics_path')

        self.create_timer(param('tick_period_sec', float), self._tick)
        if self._metrics_path:
            self.write_metrics()
            self.create_timer(5.0, self.write_metrics)
        if self.client is None:
            self.get_logger().info(
                f'Планировщик готов: mode={mode}, selection_policy=budget, model=disabled'
            )
        else:
            self.get_logger().info(
                f'LLM-планировщик готов: mode={mode}, selection_policy=llm, '
                f'model={self.client.cfg.model}, '
                f'key={"set" if self.client.cfg.configured else "missing"}'
            )

    def _tick(self) -> None:
        try:
            self.planner.tick()
        except Exception as error:  # noqa: BLE001 - the loop must not die
            self.get_logger().error(f'planning tick failed: {error}')

    def write_metrics(self) -> None:
        """Write counters atomically, including while operator control is stopped."""
        if not self._metrics_path:
            return
        path = Path(self._metrics_path)
        temporary = path.with_name(f'.{path.name}.{os.getpid()}.tmp')
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            temporary.write_text(json.dumps(self.planner.metrics(), allow_nan=False) + '\n',
                                 encoding='utf-8')
            temporary.replace(path)
        except OSError:
            self.get_logger().warning('Could not write planner metrics.')


def main(args=None) -> None:
    rclpy.init(args=args)
    node = LLMPlanner()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.write_metrics()
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
