from setuptools import find_packages
from setuptools import setup

setup(
    name='ebot_nav2',
    version='0.0.0',
    packages=find_packages(
        include=('ebot_nav2', 'ebot_nav2.*')),
)
