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

def matvecmul(mat, vec):
	"""
	Matrix-vector multiplication
	:param mat:
	:param vec:
	:return:
	"""
	return tf.linalg.LinearOperatorFullMatrix(mat).matvec(vec)
	#
	# if mat.shape.ndims == 2 and vec.shape.ndims == 1:
	# 	return tf.einsum('ij,j->i', mat, vec)
	# elif mat.shape.ndims == 2 and vec.shape.ndims == 2:
	# 	return tf.einsum('ij,aj->ai', mat, vec)
	# elif mat.shape.ndims == 3 and vec.shape.ndims == 1:
	# 	return tf.einsum('aij,j->ai', mat, vec)
	# elif mat.shape.ndims == 3 and vec.shape.ndims == 2:
	# 	return tf.einsum('aij,aj->ai', mat, vec)
	# else:
	# 	raise NotImplementedError

def matmatmul(mat1, mat2):
	"""
	Matrix-matrix multiplication
	:param mat1:
	:param mat2:
	:return:
	"""
	return tf.linalg.LinearOperatorFullMatrix(mat1).matmul(mat2)

	# if mat1.shape.ndims == 2 and mat2.shape.ndims == 2:
	# 	return tf.matmul(mat1, mat2)
	# elif mat1.shape.ndims == 3 and mat2.shape.ndims == 2:
	# 	return tf.einsum('aij,jk->aik', mat1, mat2)
	# elif mat1.shape.ndims == 2 and mat2.shape.ndims == 3:
	# 	return tf.einsum('ij,ajk->aik', mat1, mat2)
	# elif mat1.shape.ndims == 3 and mat2.shape.ndims == 3:
	# 	return tf.einsum('aij,ajk->aik', mat1, mat2)
	# else:
	# 	raise NotImplementedError

def angular_vel_tensor(w):
	if w.shape.ndims == 1:
		return tf.stack([[0., -w[2], w[1]],
						 [w[2], 0. , -w[0]],
						 [-w[1], w[0], 0.]])
	else:
		di = tf.zeros_like(w[:, 0])
		return tf.transpose(
				tf.stack([[di, -w[:, 2], w[:, 1]],
						  [w[:, 2], di , -w[:, 0]],
						  [-w[:, 1], w[:, 0], di]]),
			perm=(2, 0, 1)
		)

def drotmat_to_w_jac(r):
	"""

	:param r: rotation matrix [:, 3, 3]
	:return:
	"""
	return tf.concat([
		tf.compat.v1.matrix_transpose(angular_vel_tensor(r[:, :, i])) for i in range(3)], axis=1)

	# return tf.concat([angular_vel_tensor(tf.matrix_transpose(r)[:, :, i]) for i in range(3)], axis=1)

def rot_matrix_gains(r, k):
	"""

	r: rotation matrix [:, 3, 3]
	k: rotation matrix [:, 6, 6]
	:param r:
	:return:
	"""

	if k.shape.ndims == 2:
		k = k[None] * tf.ones_like(r[:, :1, :1])

	k_mat = tf.concat([
		tf.concat([tf.eye(3, batch_shape=(1,)) + tf.zeros_like(k[:, :3, :3])  ,
				   tf.zeros_like(k[:, :3, :3])], axis=2),
		tf.concat([tf.zeros_like(k[:, :1, :3]) * tf.ones((1, 9, 1)) ,
				   drotmat_to_w_jac(r)], axis=2)], 1)

	return tf.linalg.matmul(k, k_mat, transpose_b=True)