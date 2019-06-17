
from distutils.core import setup

import sys, platform
assert sys.version_info >= (3,3), "tensorflow_power_spectrum requires Python version >= 3.3. You have "+platform.python_version()

setup(
    name='tensorflow_power_spectrum',
    version='0.1',
    packages=['power_spectrum'],
    description='Power spectrum calculation in tensorflow',
    long_description=open('README.md').read(),
    install_requires=[
        'numpy',
        'tensorflow',
    ],
    classifiers=(
        "Programming Language :: Python :: 3.6",
        "Operating System :: OS Independent",
    ),
    
)