"""Nav2 using the agent's observed map and existing world localization.

The namespace isolates all actions and velocity commands. The agent validates
and forwards /nav2/cmd_vel only while Nav2 owns an active navigation goal.
No map server, AMCL or SLAM is started: knowledge and odom already have owners.
"""

import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch_ros.actions import Node


def generate_launch_description():
    params = os.path.join(
        get_package_share_directory('did_agent'), 'config', 'nav2.yaml')
    servers = [
        ('nav2_controller', 'controller_server'),
        ('nav2_planner', 'planner_server'),
        ('nav2_behaviors', 'behavior_server'),
        ('nav2_bt_navigator', 'bt_navigator'),
    ]
    nodes = [
        Node(
            package=package,
            executable=name,
            name=name,
            namespace='nav2',
            output='screen',
            parameters=[params],
            # TF belongs to the simulation-wide tree. cmd_vel is intentionally
            # private even if a future server uses an absolute topic name.
            remappings=[
                ('tf', '/tf'), ('tf_static', '/tf_static'),
                ('cmd_vel', '/nav2/cmd_vel'), ('/cmd_vel', '/nav2/cmd_vel'),
            ],
        )
        for package, name in servers
    ]
    nodes.append(Node(
        package='nav2_lifecycle_manager',
        executable='lifecycle_manager',
        name='lifecycle_manager_navigation',
        namespace='nav2',
        output='screen',
        parameters=[{
            'use_sim_time': True,
            'autostart': True,
            'node_names': [name for _, name in servers],
            'bond_timeout': 10.0,
        }],
    ))
    return LaunchDescription(nodes)
