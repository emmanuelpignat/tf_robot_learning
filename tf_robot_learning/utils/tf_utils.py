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
import lxml
def log_normalize(x, axis=0):
	return x - tf.reduce_logsumexp(x, axis=axis, keepdims=True)

def _outer_squared_difference(x, y):
	"""Convenience function analogous to tf.squared_difference."""
	z = x - y
	return z[..., tf.newaxis, :] * z[..., tf.newaxis]

def reduce_cov(x, axis=0, weights=None):
	assert axis == 0, NotImplementedError

	if weights is None:
		return tf.reduce_mean(_outer_squared_difference(
				tf.reduce_mean(x, axis=0, keepdims=True), x), axis=0)
	else:
		return tf.reduce_sum(weights[:, None, None] * _outer_squared_difference(
				tf.reduce_mean(x, axis=0, keepdims=True), x), axis=0)




def damped_pinv_right(J, d=1e-5):
	"""Minimizing x force"""

	s = J.shape[-1].value if not isinstance(J.shape[-1], int) else J.shape[-1]

	return tf.matmul(tf.linalg.inv(tf.matmul(J, J, transpose_a=True) + d * tf.eye(s)), J,
						 transpose_b=True)

def block_diagonal(ms):
	"""
	Create a block diagonal matrix with a list of square matrices of same sizes

	:type ms: 		lisf of tf.Tensor	[..., n_dim, n_dim]
	:return:
	"""
	import numpy as np

	n_dims = np.array([m.shape[-1].value for m in ms])

	if np.sum((np.mean(n_dims) - n_dims) ** 2):  # check if not all same dims
		return block_diagonal_different_sizes(ms)

	s = ms[0].shape[-1].value
	z = ms[0].shape.ndims - 2  # batch dims
	n = len(ms)  # final size of matrix
	mat = []

	for i, m in enumerate(ms):
		nb, na = i * s, (n - i - 1) * s
		paddings = [[0, 0] for i in range(z)] + [[nb, na], [0, 0]]
		mat += [tf.pad(m, paddings=paddings)]

	return tf.concat(mat, -1)

def block_diagonal_different_sizes(ms):
	import numpy as np
	s = np.array([m.shape[-1].value for m in ms])

	cs = [0] + np.cumsum(s).tolist()
	z = ms[0].shape.ndims - 2  # batch dims
	mat = []

	for i, m in enumerate(ms):
		nb, na = cs[i], cs[-1] - cs[i] - s[i]
		paddings = [[0, 0] for i in range(z)] + [[nb, na], [0, 0]]
		mat += [tf.pad(m, paddings=paddings)]

	return tf.concat(mat, -1)

def matquad(lin_op, m, adjoint=False):
	"""
	A^T m A
	:param lin_op:
	:type lin_op: tf.linalg.LinearOperatorFullMatrix
	:param m:
	:param adjoint : A m A ^T
	:return:
	"""
	if isinstance(lin_op, tf.Tensor):
		lin_op = tf.linalg.LinearOperatorFullMatrix(lin_op)

	if adjoint:
		return lin_op.matmul(lin_op.matmul(m), adjoint_arg=True)

	return lin_op.matmul(lin_op.matmul(m, adjoint=True), adjoint=True, adjoint_arg=True)

def matvec(lin_op, v):
	"""
	A^T v A
	:param lin_op:
	:type lin_op: tf.linalg.LinearOperatorFullMatrix
	:param v:


	:return:
	"""
	return lin_op.matvec(lin_op.matvec(v, adjoint=True), adjoint=True)

def bhatt_mvn(mvn_1, mvn_2):
	"""
	https://fr.wikipedia.org/wiki/Distance_de_Bhattacharyya	
	:param mvn_1:
	:param mvn_2:
	:return:
	"""
	P = (mvn_1.covariance() + mvn_2.covariance()) / 2.
	dloc = mvn_1.loc - mvn_2.loc

	return 1. / 8. * tf.reduce_sum(
		dloc * tf.linalg.LinearOperatorFullMatrix(
			P).solvevec(dloc), axis=-1) + \
			0.5 * (tf.linalg.slogdet(P)[1] - \
				   0.5 * (tf.linalg.slogdet(mvn_1.covariance())[1] + \
						  tf.linalg.slogdet(mvn_2.covariance())[1]))

def generalized_bhatt_mvn(mvns, reg=None):
	"""
	Never heard of this measure ?? Well, I just invented it. Deal with it. Now that you
	are here, could you check that it is actually corresponding to something ?

	:param mvns:
	:return:
	"""
	locs = tf.stack([mvn.loc for mvn in mvns])
	covs = tf.stack([mvn.covariance() for mvn in mvns])

	logdets = tf.linalg.slogdet(covs)[1]
	cov = tf.reduce_mean(covs, axis=0)

	if reg is not None:
		cov += tf.compat.v1.matrix_diag(reg ** 2. * tf.ones_like(cov[..., 0]))

	loc = tf.reduce_mean(locs, axis=0)
	dlocs = locs - tf.expand_dims(loc, axis=0)

	return 1./8. * tf.reduce_sum(
			dlocs * tf.linalg.LinearOperatorFullMatrix(
				cov).solvevec(dlocs), axis=(0, 2)) + \
		tf.reduce_sum(0.5 * (tf.linalg.slogdet(cov)[1][None] - \
			   logdets), axis=0)


def batch_jacobians(ys, xs):
	"""
	ys : [None, n_y] or [n_y]
	xs : [None, n_x] or [n_x]
	"""
	if ys.shape.ndims == 2:
		s = ys.shape[-1] if isinstance(ys.shape[-1], int) else ys.shape[-1].value
		return tf.transpose(
			tf.stack([tf.gradients(ys[:, i], xs)[0] for i in range(s)]),
			(1, 0, 2))
	elif ys.shape.ndims == 1:
		s = ys.shape[0] if isinstance(ys.shape[-1], int) else ys.shape[0].value
		return tf.stack([tf.gradients(ys[i], xs)[0] for i in range(s)])
	else:
		raise NotImplementedError

from tensorflow_probability import distributions
def reduce_mvn_ds(mvns):
	"""
	Perform moment matching
	mvns : list of mvn
	:return:
	"""

	# make h [..., 1], multiply and reduce
	locs = tf.stack([mvn.loc for mvn in mvns])
	loc = tf.reduce_mean(locs, axis=0)

	dlocs = locs - tf.expand_dims(loc, axis=0)
	cov_locs = tf.matmul(tf.expand_dims(dlocs, axis=-1),
						 tf.expand_dims(dlocs, axis=-2))


	covs = tf.stack([mvn.covariance() for mvn in mvns])
	cov = tf.reduce_mean(covs + cov_locs, axis=0)

	return distributions.MultivariateNormalFullCovariance(loc, cov)

def reduce_mvn_mm(locs=None, covs=None, h=None, axis=0):
	"""
	Perform moment matching

	:param locs: [..., n_dim]
	:param covs: [..., n_dim, n_dim]
	:param h: [...]
	:param axis:
	:return:
	"""

	if h is not None:
		# make h [..., 1], multiply and reduce
		loc = tf.reduce_sum(tf.expand_dims(h, -1) * locs, axis=axis)

		dlocs = locs - tf.expand_dims(loc, axis=axis)
		cov_locs = tf.matmul(tf.expand_dims(dlocs, axis=-1),
							 tf.expand_dims(dlocs, axis=-2))

		# make h [..., 1, 1]
		cov = tf.reduce_sum(
			tf.expand_dims(tf.expand_dims(h, -1), -1) * (covs + cov_locs), axis=axis)

	else:
		# make h [..., 1], multiply and reduce
		loc = tf.reduce_mean(locs, axis=axis)

		dlocs = locs - tf.expand_dims(loc, axis=axis)
		cov_locs = tf.matmul(tf.expand_dims(dlocs, axis=-1),
							 tf.expand_dims(dlocs, axis=-2))

		# make h [..., 1, 1]
		cov = tf.reduce_mean(covs + cov_locs, axis=axis)

	return loc, cov