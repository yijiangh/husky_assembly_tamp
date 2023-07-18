#!/usr/bin/env python
# -*- encoding: utf-8 -*-
from __future__ import absolute_import, print_function

import os
import re
import io

import argparse
from setuptools import setup, find_packages
from setuptools.command.build_ext import build_ext
# from setup_cmake_utils import CMakeExtension, CMakeBuild

here = os.path.abspath(os.path.dirname(__file__))


def read(*names, **kwargs):
    return io.open(
        os.path.join(here, *names),
        encoding=kwargs.get('encoding', 'utf8')
    ).read()


about = {}
exec(read('src', 'husky_assembly', '__version__.py'), about)

# requirements = read('requirements.txt').split('\n')
requirements = [
    'pybullet_planning>=0.6.0',
    'compas_fab==0.28.0',
    'plyer==2.1.0',
    # 'matplotlib',
]

# ext_modules = [
#     # TODO: cpp backend
#     CMakeExtension('beam_assembly'),
#     ]

setup(
    name=about['__title__'],
    version=about['__version__'],
    license=about['__license__'],
    description=about['__description__'],
    author=about['__author__'],
    author_email=about['__author_email__'],
    url=about['__url__'],
    long_description='',
    packages=find_packages('src'),
    package_dir={'': 'src'},
    # package_data={'pyconmech': ['data/*.json']},
    # ext_modules=ext_modules,
    # cmdclass=dict(build_ext=CMakeBuild),
    zip_safe=False,
    classifiers=[
        'Programming Language :: Python :: 3.8',
    ],
    keywords=[''],
    install_requires=requirements,
    # extras_require={},
    # entry_points={},
)