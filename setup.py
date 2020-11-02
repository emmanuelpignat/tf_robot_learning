# tf_robot_learning, a all-around tensorflow library for robotics.
#
# Copyright (c) 2020 Idiap Research Institute, http://www.idiap.ch/
# Written by Emmanuel Pignat <emmanuel.pignat@idiap.ch>,
#
# This file is part of tf_robot_learning.
#
# tf_robot_learning is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License version 3 as
# published by the Free Software Foundation.
#
# tf_robot_learning is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with tf_robot_learning. If not, see <http://www.gnu.org/licenses/>.

from setuptools import setup, find_packages
import sys


# Setup for Python3
setup(name='tf_robot_learning',
	  version='0.1',
	  description='Tensorflow robotic toolbox',
	  url='',
	  author='Emmanuel Pignat',
	  author_email='emmanuel.pignat@gmail.com',
	  license='MIT',
	  packages=find_packages(),
	  install_requires = ['numpy', 'matplotlib','jupyter', 'tensorflow', 'tensorflow_probability', 'pyyaml', 'lxml'],
	  zip_safe=False)
