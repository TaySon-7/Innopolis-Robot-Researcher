"""Setup for the did_llm package."""

from glob import glob
import os

from setuptools import find_packages, setup

package_name = 'did_llm'

setup(
    name=package_name,
    version='0.1.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages', ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'config'), glob('config/*.yaml')),
        (os.path.join('share', package_name, 'launch'), glob('launch/*.launch.py')),
    ],
    install_requires=['setuptools', 'pyyaml'],
    zip_safe=True,
    maintainer='DID Hack Team',
    maintainer_email='team@innopolis.local',
    description='LLM planner for the DID Hack agent',
    license='Apache-2.0',
    install_requires=['setuptools', 'pyyaml', 'pydantic>=2'],
    entry_points={
        'console_scripts': [
            'llm_planner = did_llm.llm_planner_node:main',
            'harness = did_llm.harness_node:main',
        ],
    },
)
