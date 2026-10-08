"""Launch the official TurtleBot3 world headlessly with the DID judge.

Arguments:
  scenario       easy | medium | hard (default easy)
  scenario_file  absolute path to a custom scenario YAML (overrides scenario)
"""

import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import AppendEnvironmentVariable
from launch.actions import DeclareLaunchArgument
from launch.actions import IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch.substitutions import PythonExpression
from launch_ros.actions import Node


def generate_launch_description():
    """Create the complete simulation launch description."""
    turtlebot_launch_dir = os.path.join(
        get_package_share_directory('turtlebot3_gazebo'),
        'launch',
    )
    world = os.path.join(
        get_package_share_directory('turtlebot3_gazebo'),
        'worlds',
        'turtlebot3_world.world',
    )
    models = os.path.join(
        get_package_share_directory('turtlebot3_gazebo'),
        'models',
    )
    ros_gz_sim = get_package_share_directory('ros_gz_sim')
    judge_share = get_package_share_directory('did_judge')

    scenario_file = PythonExpression([
        "'", LaunchConfiguration('scenario_file'), "' or '",
        os.path.join(judge_share, 'scenarios'), "/' + '",
        LaunchConfiguration('scenario'), ".yaml'",
    ])

    return LaunchDescription([
        DeclareLaunchArgument('scenario', default_value='easy'),
        DeclareLaunchArgument('scenario_file', default_value=''),
        AppendEnvironmentVariable('GZ_SIM_RESOURCE_PATH', models),
        IncludeLaunchDescription(
            PythonLaunchDescriptionSource(
                os.path.join(ros_gz_sim, 'launch', 'gz_sim.launch.py')
            ),
            launch_arguments={
                'gz_args': f'-r -s -v 2 {world}',
                'on_exit_shutdown': 'true',
            }.items(),
        ),
        IncludeLaunchDescription(
            PythonLaunchDescriptionSource(
                os.path.join(turtlebot_launch_dir, 'spawn_turtlebot3.launch.py')
            ),
            launch_arguments={
                'x_pose': '-2.0',
                'y_pose': '-0.5',
            }.items(),
        ),
        IncludeLaunchDescription(
            PythonLaunchDescriptionSource(
                os.path.join(
                    turtlebot_launch_dir,
                    'robot_state_publisher.launch.py',
                )
            ),
            launch_arguments={'use_sim_time': 'true'}.items(),
        ),
        Node(
            package='did_judge',
            executable='judge',
            name='judge',
            output='screen',
            parameters=[{
                'use_sim_time': True,
                'scenario_file': scenario_file,
            }],
        ),
    ])

