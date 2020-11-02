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

import tensorflow as tf
from tensorflow_probability import distributions as ds
from ..distributions import *
from .mlp import MLP
import numpy as np



param_size = {
	MultivariateNormalTriL : lambda x : [x,  x ** 2],
	MultivariateNormalFullCovariance: lambda x : [x,  x ** 2],
	MultivariateNormalDiag : lambda x : [x,  x],
	MultivariateNormalIso : lambda x : [x,  1],
}

param_shape = {
	MultivariateNormalTriL : lambda x : [(x, ),  (x, x)],
	MultivariateNormalFullCovariance : lambda x : [(x, ),  (x, x)],
	MultivariateNormalDiag : lambda x : [(x, ),  (x, )],
	MultivariateNormalIso : lambda x : [(x, ),  (1, )]
}

reparam = {
	MultivariateNormalTriL :
		[ lambda x : x,  lambda x : x],
	MultivariateNormalFullCovariance:
		[ lambda x : x,  lambda x : tf.linalg.expm(x + tf.matrix_transpose(x))],
	MultivariateNormalDiag :
		[ lambda x : x,  lambda x : tf.exp(x)],
	MultivariateNormalIso :
		[ lambda x : x,  lambda x : tf.exp(x)]
}


class DensityMLP(MLP):
	def __init__(
			self, n_input, n_output, n_hidden=[12, 12], batch_size_svi=1,
			act_fct=tf.nn.relu, density=MultivariateNormalTriL):


			self._density = density

			self._n_output_event = n_output

			# compute the shape of the parameters of the distribution
			assert density in param_size, "Parameters shape not defined for specified distribution"

			n_output_param = sum(param_size[density](n_output))

			MLP.__init__(
				self, n_input, n_output_param, n_hidden=n_hidden,
				batch_size_svi=batch_size_svi, act_fct=act_fct
			)

	def density(self, x, vec_weights=None):
		# get parameters of conditional distribution

		vec_params = self.pred(x, vec_weights=vec_weights, avoid_concat=True)

		# vec_params is of shape (batch_size_svi, batch_size_data, n_output_param)
		_param_size = param_size[self._density](self._n_output_event)
		_param_shape = param_shape[self._density](self._n_output_event)
		_reparam = reparam[self._density]
		# get slices for each parameters
		slices = [slice(s, e) for s, e in zip(
			np.cumsum([0] + _param_size),
			np.cumsum(_param_size)
		)]

		params = [
			r(tf.reshape(vec_params[:, :, sl], (self._batch_size_svi, -1,) + s ))
			for sl, s, r in zip(slices, _param_shape, _reparam)
		]

		if self._batch_size_svi == 1:
			params = [p[0] for p in params]


		return self._density(*params)




