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

import numpy as np
from .tf_utils import block_diagonal


def build_fixed_psi(n_step, n_dim, n_state, scale=None, dtype=tf.float32):
	_hs, _psis, _ = build_psis([n_step], n_state, n_dim, scale=scale)

	h = tf.constant(_hs[0].eval())
	psi = tf.constant(_psis[0].eval())

	return psi, h


def build_psi_partial(h, n_dim, dim_obs, n_step=None, n_state=None, dtype=tf.float32):
	"""
	For conditioning, build psi with partial observation

	:param dim_obs:
	:type dim_obs: list or slice
	:param h: [n_timestep, n_state]
	:return:
	"""
	if isinstance(dim_obs, slice):
		dim_obs = [i for i in range(dim_obs.start, dim_obs.stop)]

	n_dim_in = len(dim_obs)

	n_dim_traj = n_dim_in * tf.shape(h)[0] if n_step is None else n_dim_in * n_step
	n_state = tf.shape(h)[1] if n_state is None else n_state

	obs_matrix = np.zeros((n_dim_in, n_dim))

	for j, i in enumerate(dim_obs):
		obs_matrix[j, i] = 1.

	obs_matrix = tf.convert_to_tensor(obs_matrix, dtype=dtype)

	return tf.transpose(tf.reshape(
		tf.transpose(h[:, :, None, None] * obs_matrix[None, None],
					 perm=(0, 2, 1, 3)),
		shape=(n_dim_traj, n_dim * n_state)))

def build_psi(h, n_dim, n_step=None, n_state=None, dtype=tf.float32):
	"""

	:param h: [n_timestep, n_state]
	:return:
	"""
	n_dim_traj = n_dim * tf.shape(h)[0] if n_step is None else n_dim * n_step
	n_state = tf.shape(h)[1] if n_state is None else n_state

	return tf.transpose(tf.reshape(
		tf.transpose(h[:, :, None, None] * tf.eye(n_dim, dtype=dtype)[None, None],
								   perm=(0, 2, 1, 3)),
                        shape=(n_dim_traj, n_dim * n_state)))

def build_psis(n_steps, n_state, n_dim, scale=None, fixed_handles=True, dtype=tf.float32,
			   dim_obs=None):
	"""

	:param n_steps:			Length of each trajectory
	:type n_steps: 		list of int
	:param n_state:			Number of basis function
	:type n_state: 		int
	:type scale: 	float 0. - 1.
		:return: psis, hs, handles
	else:
		:return: psis, hs

	"""
	from tensorflow_probability import distributions as ds

	# create handles for each demos
	if fixed_handles:
		handles = tf.stack([tf.linspace(tf.cast(0., dtype=dtype), tf.cast(n, dtype=dtype), n_state)
							for n in n_steps])
	else:
		handles = tf.Variable([tf.linspace(tf.cast(0., dtype=dtype), tf.cast(n, dtype=dtype), n_state)
							   for n in n_steps])

	n_traj = len(n_steps)

	if scale is None:
		scale = 1./ n_state

	# create mixtures whose p_z will be the activations



	h_mixture = [
		ds.MixtureSameFamily(
			mixture_distribution=ds.Categorical(logits=tf.ones(n_state)),
			components_distribution=ds.MultivariateNormalDiag(
				loc=handles[j][:, None],
				scale_diag=tf.cast(scale, dtype)[None, None] * n_steps[j]
			)
		)
		for j in range(n_traj)]


	# create evaluation points of the mixture for each demo
	idx = [tf.range(n, dtype=dtype) for n in n_steps]
	from .tf_utils import log_normalize
	j = 0
	# create activations
	# print tf.transpose(h_mixture[0].components_log_prob(idx[0][:, None]))
	hs = [tf.exp(log_normalize(
		h_mixture[j].components_distribution.log_prob(idx[j][:, None, None]), axis=1))
		for j in range(n_traj)]
	# hs = [tf for h in hs]
	if dim_obs is None:
		psis = [build_psi(hs[i], n_dim, n, n_state, dtype=dtype)
				for i, n in enumerate(n_steps)]
	else:
		psis = [build_psi_partial(hs[i], n_dim, dim_obs, n, n_state, dtype=dtype)
				for i, n in enumerate(n_steps)]

	return hs, psis, handles


def build_obs(y):
	"""

	:param y:
	:type y: 		tf.Tensor   [..., n_timestep, n_dim]
	:return:
	"""
	if y.shape.ndims == 3:
		return tf.reshape(y, (tf.shape(y)[0], -1))
	elif y.shape.ndims == 2:
		return tf.reshape(y, (-1,))


def build_promp_transform(A, b):
	"""
	Build A, b such to move from the object coordinate system to the parent

	:param A:	Rotation matrix of the object
	:type A: 			tf.Tensor or np.ndarray	[..., n_dim, n_dim]
	:param b:	Position of the object
	:type b: 			tf.Tensor or np.ndarray	[..., n_dim]
	:return:
	"""

	b = tf.reshape(tf.transpose(b, (1, 0, 2)), (b.shape[1].value, -1))

	A = block_diagonal([A[i] for i in range(A.shape[0].value)])

	return A, b


def build_dct_matrix(m, n_comp=None,
                     dtype=tf.float32, name='dct'):
	"""

	:param m:
	:param dtype:
	:param name:
	:return:
	"""

	if n_comp is None:
		n_comp = m

	dctm = np.zeros((n_comp, m))
	sqrt_m = np.sqrt(m)

	for p in range(n_comp):
		for q in range(m):
			if p == 0:
				dctm[p, q] = 1/sqrt_m
			elif p > 0:
				dctm[p, q] = 2**0.5/sqrt_m * np.cos(np.pi * (2 * q + 1) * p/2./m )

	return tf.convert_to_tensor(dctm, dtype=dtype, name=name)


def seq_dct(samples, dct_m, inverse=False):
	"""
	samples : [batch_size, timesteps, xi_dim]
	"""
	return tf.transpose(dct_m.matvec(tf.transpose(samples, (0, 2, 1)), adjoint=inverse), (0, 2, 1))

def seq_lp(samples):
	return tf.log(samples ** 2)

def correlated_normal_noise(shape, dct_components, mean=0., scale=1.):
	"""
	shape, (batch_size, horizon, xi_dim)
	"""
	k_basis = dct_components.shape[0] if isinstance(dct_components.shape[0], int) else \
	dct_components.shape[0].value

	_, h = build_fixed_psi(
		n_step=shape[1], n_dim=shape[-1], n_state=k_basis, scale=0.5 / k_basis)

	dct_m = tf.linalg.LinearOperatorFullMatrix(build_dct_matrix(shape[1]))

	noise = tf.random.normal(shape, mean, scale)
	dct_components_t = tf.linalg.matvec(h, dct_components)
	dct_norm_t = dct_components_t / tf.linalg.norm(dct_components_t) * shape[1] ** 0.5
	return seq_dct(seq_dct(noise, dct_m) * dct_norm_t[None, :, None], dct_m, inverse=True)