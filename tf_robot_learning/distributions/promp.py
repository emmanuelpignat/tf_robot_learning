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
from ..utils.basis_utils import build_fixed_psi


class ProMP(ds.MultivariateNormalFullCovariance):
	def __init__(self,
				 loc=None,
				 covariance_matrix=None,
				 scale_obs=0.1,
				 psi=None,
				 fast_sample=True,
				 validate_args=False,
				 allow_nan_stats=True,
				 name="ProMP"):

		self._psi = psi
		self._psi_op = tf.linalg.LinearOperatorFullMatrix(psi)
		self._loc_w = loc
		self._cov_w = covariance_matrix

		self._mvn_w = ds.MultivariateNormalFullCovariance(
			loc=self._loc_w,
			covariance_matrix=self._cov_w
		)

		# (psi T cov psi ) T = (psi T cov )T

		# _loc = tf.linalg.matvec(psi, loc, transpose_a=True)
		# _cov = tf.linalg.matmul(tf.linalg.matmul(psi, covariance_matrix, transpose_a=True),
		# 						psi) + \
			   # tf.eye(psi.shape[1].value) * scale_obs ** 2

		_loc = self._psi_op.matvec(loc, adjoint=True)

		_cov =  self._psi_op.matmul(
			self._psi_op.matmul(covariance_matrix, adjoint=True), adjoint=True, adjoint_arg=True)

		self._mvn_obs = ds.MultivariateNormalDiag(
			loc=tf.zeros_like(_loc),
			scale_diag=scale_obs * tf.ones(psi.shape[1].value),
		)

		self._fast_sample = fast_sample

		if _cov.shape.ndims == 2:
			_cov += tf.eye(psi.shape[1].value) * scale_obs ** 2
		elif _cov.shape.ndims == 3:
			_cov += tf.eye(psi.shape[1].value)[None] * scale_obs ** 2
		else:
			raise NotImplementedError

		ds.MultivariateNormalFullCovariance.__init__(
			self, loc=_loc, covariance_matrix=_cov)

	def sample(self, sample_shape=(), seed=None, name="sample"):
		if self._fast_sample:
			_w_samples = self._mvn_w.sample(sample_shape, seed=seed, name=name + '_w')
			_obs_samples = self._mvn_obs.sample(sample_shape, seed=seed, name=name + '_obs')

			return self._psi_op.matvec(_w_samples, adjoint=True) + _obs_samples
		else:
			return ds.MultivariateNormalFullCovariance.sample(
				self, sample_shape=sample_shape, seed=seed, name=name)
