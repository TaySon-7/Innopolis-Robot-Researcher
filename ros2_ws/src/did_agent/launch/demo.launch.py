"""Everything for the demo: simulation, judge, agent and the web dashboard.

Arguments:
  scenario       easy | medium | hard, or difficulty@seed (e.g. hard@7) to generate
                 a new scenario on the fly (default easy)
  scenario_file  absolute path to a custom scenario YAML (overrides scenario);
                 'none' or empty means: use `scenario`
  navigation     custom | nav2; both backends can be selected at runtime
"""

import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.actions import IncludeLaunchDescription
from launch.actions import OpaqueFunction
from launch.conditions import IfCondition
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
    """Create the demo launch description.

    The LLM planner is the default high-level controller and publishes plans
    to /agent/plan. It falls back to the agent's deterministic policy if the
    model is not configured or is unreachable. Pass ``llm:=false`` for a
    deliberately offline run without the planner node.
    """
    return LaunchDescription([
        DeclareLaunchArgument('scenario', default_value='easy'),
        DeclareLaunchArgument('scenario_file', default_value=''),
        DeclareLaunchArgument(
            'navigation', default_value='custom', choices=['custom', 'nav2'],
            description='Initial low-level navigation backend.'),
        DeclareLaunchArgument(
            'llm', default_value='true',
            description='Start the LLM planner alongside the agent.'),
        OpaqueFunction(function=_simulation),
        # Nav2 stays available even when custom navigation is selected. It
        # publishes only private commands; the agent owns the /cmd_vel gate.
        IncludeLaunchDescription(
            PythonLaunchDescriptionSource(os.path.join(
                get_package_share_directory('did_agent'),
                'launch', 'nav2.launch.py')),
        ),
        Node(
            package='ros_gz_bridge',
            executable='parameter_bridge',
            name='gazebo_restart_bridge',
            output='screen',
            arguments=[
                '/world/default/remove/blocking@ros_gz_interfaces/srv/DeleteEntity',
                '/world/default/create/blocking@ros_gz_interfaces/srv/SpawnEntity',
            ],
        ),
        Node(
            package='did_agent', executable='agent', name='agent', output='screen',
            parameters=[{'navigation_backend': LaunchConfiguration('navigation')}],
        ),
        Node(package='did_agent', executable='dashboard', name='dashboard', output='screen'),
        Node(
            package='rosbridge_server',
            executable='rosbridge_websocket',
            name='rosbridge_websocket',
            output='screen',
            parameters=[{'port': 9090}],
        ),
        Node(
            package='did_llm',
            executable='llm_planner',
            name='llm_planner',
            output='screen',
            condition=IfCondition(LaunchConfiguration('llm')),
            # Sim time, like every other node here: Gazebo drives /clock, and a
            # planner measuring wall time would think the episode is paused
            # whenever the simulation lags.
            parameters=[os.path.join(
                get_package_share_directory('did_llm'),
                'config', 'planner.yaml'),
                {'use_sim_time': True}],
        ),
    ])
