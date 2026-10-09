from glob import glob
import os

from setuptools import find_packages
from setuptools import setup


package_name = 'did_agent'


setup(
    name=package_name,
    version='0.1.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        (
            'share/ament_index/resource_index/packages',
            ['resource/' + package_name],
        ),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'config'), glob('config/*.yaml')),
        (os.path.join('share', package_name, 'maps'), glob('maps/*')),
        (os.path.join('share', package_name, 'web'), glob('web/*')),
        (os.path.join('share', package_name, 'launch'), glob('launch/*.launch.py')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='DID team',
    maintainer_email='team@example.com',
    description='Navigation, skills and executor of the Robot Researcher agent.',
    license='Apache-2.0',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'goto = did_agent.nav_node:main',
            'agent = did_agent.agent_node:main',
            'auto = did_agent.agent_node:main_auto',
            'send_plan = did_agent.agent_node:main_send_plan',
            'command = did_agent.agent_node:main_command',
            'dashboard = did_agent.dashboard_node:main',
            'generate_scenario = did_agent.scenario_generator:main',
        ],
    },
)
