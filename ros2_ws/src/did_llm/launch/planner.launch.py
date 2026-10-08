#!/usr/bin/env python3
"""Launch the LLM planner alongside a running stand.

Kept separate from ``did_agent.demo.launch.py`` on purpose: the planner needs
the agent to be up first, and starting it in a second terminal also means the
demo degrades to the agent's own behaviour simply by not starting it.

    ros2 launch did_llm planner.launch.py
    ros2 launch did_llm planner.launch.py use_sim_time:=false
"""

import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node

CONFIG_DEFAULT = os.path.join(
    get_package_share_directory('did_llm'), 'config', 'planner.yaml')


def generate_launch_description():
    config = LaunchConfiguration('params_file')
    use_sim_time = LaunchConfiguration('use_sim_time')

    return LaunchDescription([
        DeclareLaunchArgument(
            'params_file', default_value=CONFIG_DEFAULT,
            description='Planner configuration file.'),
        DeclareLaunchArgument('use_sim_time', default_value='true'),

        Node(
            package='did_llm',
            executable='llm_planner',
            name='llm_planner',
            output='screen',
            # use_sim_time goes inside the parameters list: a YAML file cannot
            # be mixed with a dict that way, and Jazzy's Node has no
            # additional_params argument.
            parameters=[config, {'use_sim_time': use_sim_time}],
        ),
    ])