# -*- coding: utf-8 -*-

# Learn more: https://github.com/kennethreitz/setup.py

from setuptools import setup, find_packages


with open('README.rst') as f:
    readme = f.read()

with open('LICENSE') as f:
    license = f.read()

setup(
    name='das_workflows',
    version='1.0.0',
    description='Package for development of turbogap workflows in DAS group @aalto',
    long_description=readme,
    author='Jan Kloppenburg',
    author_email='jank@numphys.org',
    url='https://github.com/janklinux/das_workflows',
    license=license,
    packages=find_packages(exclude=('tests', 'docs'))
)

