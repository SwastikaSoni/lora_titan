import os
from glob import glob
from setuptools import setup, find_packages

package_name = 'titan_coordination'

setup(
    name=package_name,
    version='0.1.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        # Install launch files
        (os.path.join('share', package_name, 'launch'),
            glob('launch/*.py')),
        # Install config files (bridge.yaml etc.)
        (os.path.join('share', package_name, 'config'),
            glob('config/*.yaml')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='Student',
    maintainer_email='student@nitw.ac.in',
    description='Platoon coordination for Titan DMS',
    license='MIT',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'platoon_leader = titan_coordination.platoon_leader:main',
            'mesh_bridge_node = titan_coordination.mesh_bridge_node:main',
            'bs_sink_node = titan_coordination.bs_sink_node:main',
            'platoon_follower = titan_coordination.platoon_follower:main',
        ],
    },
)