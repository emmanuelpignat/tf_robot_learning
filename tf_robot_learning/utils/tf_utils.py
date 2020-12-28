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

# from tensorflow.python.ops.parallel_for.gradients import batch_jacobian

def nullspace_transformation(x=None, f=None, J=None, d=1e-5):
	"""
	Get nullspace filter of a transformation where dim_in > dim_out

	:param q: 	Tensor [batch_size, dim_in]
	:param f:	function [batch_size, dim_in] -> [batch_size, dim_out]
	:return:	nullspace [batch_size, dim_in, dim_in]
	"""
	if J is None:
		y = f(x)
		J = batch_jacobian(y, x)
		size = x.shape[-1]

	else:
		size = J.shape[-1]

	return tf.eye(size)[None] - tf.matmul(damped_pinv_right(J, d=d), J)


def nullspace_project(fct, f_main, x, x_size=None):
	"""
	Project gradient of f in nullspace of f_main, return y and gradient
	for redefining gradient

	example of usage :

		@tf.custom_gradient
		def f1_filtered(q):
			return nullspace_project(f1, f0, x)


	:param f:		function [batch_size, dim_in] -> [batch_size, dim_out_1]
			or a list of function
	:param f_main :	function [batch_size, dim_in] -> [batch_size, dim_out_2]
	:param x:		Tensor [batch_size, dim_in]
	:param x_size : int
	:return:
	"""
	if x_size is not None:
		x = tf.reshape(x, (-1, x_size))

	def grad(dx):
		J = batch_jacobian(fct(x), x)

		if not isinstance(f_main, list):
			J = tf.einsum(
				'aij,ajk->aik', J, tf.stop_gradient(
					nullspace_transformation(x=x, f=lambda x: f_main(x))))
		else:
			for _f_main in f_main:
				J = tf.einsum(
					'aij,ajk->aik', J, tf.stop_gradient(
						nullspace_transformation(x=x, f=lambda x: _f_main(x))))

		return tf.einsum('ai,aij->aj', dx, J)

	return fct(x), grad
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

# Copyright 2018 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import check_ops
from tensorflow.python.ops import gradients_impl as gradient_ops
from tensorflow.python.ops.parallel_for import control_flow_ops
from tensorflow.python.util import nest


def jacobian(output, inputs, use_pfor=True, parallel_iterations=None):
	"""Computes jacobian of `output` w.r.t. `inputs`.

	Args:
	  output: A tensor.
	  inputs: A tensor or a nested structure of tensor objects.
	  use_pfor: If true, uses pfor for computing the jacobian. Else uses
		tf.while_loop.
	  parallel_iterations: A knob to control how many iterations and dispatched in
		parallel. This knob can be used to control the total memory usage.

	Returns:
	  A tensor or a nested structure of tensors with the same structure as
	  `inputs`. Each entry is the jacobian of `output` w.r.t. to the corresponding
	  value in `inputs`. If output has shape [y_1, ..., y_n] and inputs_i has
	  shape [x_1, ..., x_m], the corresponding jacobian has shape
	  [y_1, ..., y_n, x_1, ..., x_m]. Note that in cases where the gradient is
	  sparse (IndexedSlices), jacobian function currently makes it dense and
	  returns a Tensor instead. This may change in the future.
	"""
	flat_inputs = nest.flatten(inputs)
	output_tensor_shape = output.shape
	output_shape = array_ops.shape(output)
	output = array_ops.reshape(output, [-1])

	def loop_fn(i):
		y = array_ops.gather(output, i)
		return gradient_ops.gradients(y, flat_inputs)

	try:
		output_size = int(output.shape[0])
	except TypeError:
		output_size = array_ops.shape(output)[0]

	if use_pfor:
		pfor_outputs = control_flow_ops.pfor(
			loop_fn, output_size, parallel_iterations=parallel_iterations)
	else:
		pfor_outputs = control_flow_ops.for_loop(
			loop_fn,
			[output.dtype] * len(flat_inputs),
			output_size,
			parallel_iterations=parallel_iterations)

	for i, out in enumerate(pfor_outputs):
		if isinstance(out, ops.Tensor):
			new_shape = array_ops.concat(
				[output_shape, array_ops.shape(out)[1:]], axis=0)
			out = array_ops.reshape(out, new_shape)
			out.set_shape(output_tensor_shape.concatenate(flat_inputs[i].shape))
			pfor_outputs[i] = out

	return nest.pack_sequence_as(inputs, pfor_outputs)


def batch_jacobian(output, inp, use_pfor=True, parallel_iterations=None):
	"""Computes and stacks jacobians of `output[i,...]` w.r.t. `input[i,...]`.

	e.g.
	x = tf.constant([[1, 2], [3, 4]], dtype=tf.float32)
	y = x * x
	jacobian = batch_jacobian(y, x)
	# => [[[2,  0], [0,  4]], [[6,  0], [0,  8]]]

	Args:
	  output: A tensor with shape [b, y1, ..., y_n]. `output[i,...]` should
		only depend on `inp[i,...]`.
	  inp: A tensor with shape [b, x1, ..., x_m]
	  use_pfor: If true, uses pfor for computing the Jacobian. Else uses a
		tf.while_loop.
	  parallel_iterations: A knob to control how many iterations are vectorized
		and dispatched in parallel. The default value of None, when use_pfor is
		true, corresponds to vectorizing all the iterations. When use_pfor is
		false, the default value of None corresponds to parallel_iterations=10.
		This knob can be used to control the total memory usage.

	Returns:
	  A tensor `t` with shape [b, y_1, ..., y_n, x1, ..., x_m] where `t[i, ...]`
	  is the jacobian of `output[i, ...]` w.r.t. `inp[i, ...]`, i.e. stacked
	  per-example jacobians.

	Raises:
	  ValueError: if first dimension of `output` and `inp` do not match.
	"""
	output_shape = output.shape
	# if not output_shape[0].is_compatible_with(inp.shape[0]):
	#   raise ValueError("Need first dimension of output shape (%s) and inp shape "
	#                    "(%s) to match." % (output.shape, inp.shape))

	if output_shape.is_fully_defined():
		batch_size = int(output_shape[0])
		output_row_size = output_shape.num_elements() // batch_size
	else:
		output_shape = array_ops.shape(output)
		batch_size = output_shape[0]
		output_row_size = array_ops.size(output) // batch_size
	inp_shape = array_ops.shape(inp)
	# Flatten output to 2-D.
	with ops.control_dependencies(
			[check_ops.assert_equal(batch_size, inp_shape[0])]):
		output = array_ops.reshape(output, [batch_size, output_row_size])

	def loop_fn(i):
		y = array_ops.gather(output, i, axis=1)
		return gradient_ops.gradients(y, inp)[0]

	if use_pfor:
		pfor_output = control_flow_ops.pfor(loop_fn, output_row_size,
											parallel_iterations=parallel_iterations)
	else:
		pfor_output = control_flow_ops.for_loop(
			loop_fn, output.dtype,
			output_row_size,
			parallel_iterations=parallel_iterations)
	if pfor_output is None:
		return None
	pfor_output = array_ops.reshape(pfor_output,
									[output_row_size, batch_size, -1])
	output = array_ops.transpose(pfor_output, [1, 0, 2])
	new_shape = array_ops.concat([output_shape, inp_shape[1:]], axis=0)
	return array_ops.reshape(output, new_shape)
