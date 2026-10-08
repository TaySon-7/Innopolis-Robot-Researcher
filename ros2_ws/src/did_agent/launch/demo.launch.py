"""Everything for the demo: simulation, judge, agent and the web dashboard.

Arguments:
  scenario       easy | medium | hard, or difficulty@seed (e.g. hard@7) to generate
                 a new scenario on the fly (default easy)
  scenario_file  absolute path to a custom scenario YAML (overrides scenario);
                 'none' or empty means: use `scenario`
"""

import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.actions import IncludeLaunchDescription
from launch.actions import OpaqueFunction
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def _simulation(context):
    """Include the simulation; generate the scenario first if it is name@seed."""
    scenario = LaunchConfiguration('scenario').perform(context)
    scenario_file = LaunchConfiguration('scenario_file').perform(context)
    if scenario_file == 'none':  # compose cannot pass an empty launch argument
        scenario_file = ''
    if not scenario_file and '@' in scenario:
        from did_agent.scenario_generator import load_named
        from did_agent.scenario_generator import to_yaml
        generated = load_named(scenario)
        os.makedirs('/tmp/scenarios', exist_ok=True)
        scenario_file = f"/tmp/scenarios/{scenario.replace('@', '-')}.yaml"
        with open(scenario_file, 'w', encoding='utf-8') as handle:
            handle.write(to_yaml(generated))
    judge_share = get_package_share_directory('did_judge')
    return [
        IncludeLaunchDescription(
            PythonLaunchDescriptionSource(
                os.path.join(judge_share, 'launch', 'sim.launch.py')
            ),
            launch_arguments={
                'scenario': scenario,
                'scenario_file': scenario_file,
            }.items(),
        ),
    ]


def generate_launch_description():
    """Create the demo launch description."""
    return LaunchDescription([
        DeclareLaunchArgument('scenario', default_value='easy'),
        DeclareLaunchArgument('scenario_file', default_value=''),
        OpaqueFunction(function=_simulation),
        Node(package='did_agent', executable='agent', name='agent', output='screen'),
        Node(package='did_agent', executable='dashboard', name='dashboard', output='screen'),
    ])
