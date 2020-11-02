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

from tensorflow_probability import distributions as _distributions

from .poe import PoE

from .mvn import MultivariateNormalFullCovarianceML, MultivariateNormalFullPrecision, \
	MultivariateNormalIso, MultivariateNormalFullCovariance
from .soft_uniform import SoftUniformNormalCdf, SoftUniform
# import from tensorflow_probability
from . import approx
from .promp import ProMP, build_fixed_psi

Categorical = _distributions.Categorical
MultivariateNormalDiag = _distributions.MultivariateNormalDiag
MultivariateNormalTriL = _distributions.MultivariateNormalTriL
try:
	Wishart = _distributions.Wishart
except:
	Wishart = None
LogNormal = _distributions.LogNormal
StudentT = _distributions.StudentT
Normal = _distributions.Normal
Uniform = _distributions.Uniform
MixtureSameFamily = _distributions.MixtureSameFamily
TransformedDistribution = _distributions.TransformedDistribution
Mixture = _distributions.Mixture

from .mixture_models import *