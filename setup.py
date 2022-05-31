#!/usr/bin/env python
# -*- encoding: utf-8 -*-
from __future__ import absolute_import
from setuptools import find_packages, setup


setup(
    name='interpmcmc',
    version='0.0.0',
    license='MIT',
    description='Utitilitise for running MCMC on a potential defined by interpolation',
    author='Erik Henning Thiede, ...',
    author_email='ehthiede@gmail.com',
    # url='https://github.com/risilab/Autobahn',
    url='',
    packages=find_packages('src'),
    package_dir={'': 'src'},
    include_package_data=True,
    zip_safe=False,
    python_requires='>=3.8',
)
