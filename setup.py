#!/bin/env python
from setuptools import setup, find_packages

setup(
    name='ml_utils',
    version='0.1.0',
    author='Linn Abraham',
    author_email='linn.official@gmail.com',
    description='A python package for utility scripts and functions for machine learning applications',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    url='https://github.com/linnabraham/ml-scripts',  # Update with your repository URL
    packages=find_packages(),  # Automatically finds your package
    license="MIT",
    classifiers=[
        'Programming Language :: Python :: 3',
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.6',
)
