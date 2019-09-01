#!/usr/bin/env python

from distutils.core import setup, Command

try:  # Python 3.x
    from distutils.command.build_py import build_py_2to3 as build_py
except ImportError:  # Python 2.x
    from distutils.command.build_py import build_py

setup(name='astrotools',
      version='1.0',
      description='A library of useful tools for Astronomy',
      author='Jan-Torge Schindler',
      author_email='jtschi@posteo.net',
      license='GPL',
      url='http://github.com/jtschindler/astrotools',
      packages=['astrotools', 'astrotools/speconed'],
      provides=['astrotools'],
      package_dir={'astrotools':'astrotools'},
      package_data={'astrotools':['data/*.*','data/passbands/*.*',
                                  'data/flux_standards/*.*',
                                  'data/iron_templates/*.*']},
      requires=['numpy', 'matplotlib', 'scipy', 'astropy', 'pandas'],
      keywords=['Scientific/Engineering'],
     )
