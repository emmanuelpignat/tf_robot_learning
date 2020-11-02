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

import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
plt.style.use('ggplot')


class DensityPlotter(object):
	def __init__(self, f, nb_sub=20, np=False, vectorized=True):
		if not np:
			self.x_batch = tf.compat.v1.placeholder(tf.float32, (nb_sub ** 2, 2))
			self.fx_batch = f(self.x_batch)

		self.f = f
		self.nb_sub = nb_sub
		self.vmin = 0.
		self.vmax = 0.

		self.vectorized = vectorized

		self.np = np

	def f_np(self, x, feed_dict={}):
		if self.np:
			return self.f(x)
		else:
			fd = {self.x_batch: x}
			fd.update(feed_dict)
			return self.fx_batch.eval(fd)

	def plot(self, ax0=None, xlim=[-1, 1], ylim=[-1, 1], cmap='viridis',
				 lines=True, heightmap=True, inv=False, vmin=None, vmax=None, use_tf=False,
				 ax=None, feed_dict={}, exp=True, img=False, act_fct=None, kwargs_lines={}):

		z = plot_density(lambda x: self.f_np(x, feed_dict=feed_dict),
							nb_sub=self.nb_sub, ax0=ax0, xlim=xlim, ylim=ylim, cmap=cmap,
							lines=lines, heightmap=heightmap, inv=inv, vmin=vmin, vmax=vmax, use_tf=False,
							ax=ax, feed_dict=feed_dict, exp=exp, img=img, kwargs_lines=kwargs_lines,
						 vectorized=self.vectorized, act_fct=act_fct)

		if exp:
			self.vmax, self.vmin = np.max(np.exp(z)), np.min(np.exp(z))
		else:
			self.vmax, self.vmin = np.max(z), np.min(z)


		return z


def plot_density(f, nb_sub=10, ax0=None, xlim=[-1, 1], ylim=[-1, 1], cmap='viridis',
				 lines=True, heightmap=True, inv=False, vmin=None, vmax=None, use_tf=False,
				 ax=None, feed_dict={}, exp=True, img=False, kwargs_lines={}, vectorized=True,
				 act_fct=None):

	if use_tf:
		x_batch = tf.placeholder(tf.float32, (nb_sub ** 2, 2))
		fx_batch = f(x_batch)
		def f_np(x):
			fd = {x_batch: x}
			fd.update(**feed_dict)
			return fx_batch.eval(fd)

		f = f_np


	x = np.linspace(*xlim, num=nb_sub)
	y = np.linspace(*ylim, num=nb_sub)

	if img:
		Y, X = np.meshgrid(x, y)
	else:
		X, Y = np.meshgrid(x, y)

	if vectorized:
		zs = f(np.concatenate([np.atleast_2d(X.ravel()), np.atleast_2d(Y.ravel())]).T)

	else:
		_xys = np.concatenate([np.atleast_2d(X.ravel()), np.atleast_2d(Y.ravel())]).T
		zs = []
		for _xy in _xys:
			try:
				zs += [f(_xy)]
			except:
				zs += [0.]

		zs = np.array(zs)

	if act_fct is not None:
		zs = act_fct(zs)

	Z = zs.reshape(X.shape)


	if lines:
		if ax is None:
			plt.contour(X, Y, Z, **kwargs_lines)
		else:
			ax.contour(X, Y, Z, **kwargs_lines)


	if heightmap:
		if exp:
			_img = np.exp(Z) if not inv else -np.exp(Z)
		else:
			_img = Z if not inv else -Z

		kwargs = {'origin': 'lower'}

		if ax is None:
			plt.imshow(_img, interpolation='bilinear', extent=xlim + ylim,
				   alpha=0.5, cmap=cmap, vmax=vmax, vmin=vmin, **kwargs)
		else:
			ax.imshow(_img, interpolation='bilinear', extent=xlim + ylim,
				   alpha=0.5, cmap=cmap, vmax=vmax, vmin=vmin, **kwargs)

	return Z


class PolicyPlot(object):
	def __init__(self, pi, nb_sub=20):

		self._x = tf.compat.v1.placeholder(tf.float32, (nb_sub**2, 2))
		self._u = tf.transpose(pi(self._x))
		self._nb_sub = nb_sub
	def plot(self, ax=None, xlim=[-1, 1], ylim=[-1, 1], scale=0.01,
							name=None, equal=False, feed_dict={}, sess=None, **kwargs):
		"""
		Plot a dynamical system dx = f(x)
		:param f: 		a function that takes as input x as [N,2] and return dx [N, 2]
		:param nb_sub:
		:param ax0:
		:param xlim:
		:param ylim:
		:param scale:
		:param kwargs:
		:return:
		"""

		if sess is None:
			sess = tf.compat.v1.get_default_session()

		Y, X = np.mgrid[
			   ylim[0]:ylim[1]:complex(self._nb_sub),
			   xlim[0]:xlim[1]:complex(self._nb_sub)]
		mesh_data = np.vstack([X.ravel(), Y.ravel()])

		feed_dict[self._x] = mesh_data.T
		field = sess.run(self._u, feed_dict)

		U = field[0]
		V = field[1]
		U = U.reshape(self._nb_sub, self._nb_sub)
		V = V.reshape(self._nb_sub, self._nb_sub)
		speed = np.sqrt(U * U + V * V)

		if name is not None:
			plt.suptitle(name)

		if ax is not None:
			strm = ax.streamplot(X, Y, U, V, linewidth=scale * speed, **kwargs)
			ax.set_xlim(xlim)
			ax.set_ylim(ylim)

			if equal:
				ax.set_aspect('equal')

		else:
			strm = plt.streamplot(X, Y, U, V, linewidth=scale * speed, **kwargs)
			plt.xlim(xlim)
			plt.ylim(ylim)

			if equal:
				plt.axes().set_aspect('equal')

		return [strm]



class MVNPlot(object):
	def __init__(self, ds, nb_segments=20):

		from ..distributions import GaussianMixtureModelML, GaussianMixtureModelFromSK
		self._ds = ds
		if isinstance(ds, GaussianMixtureModelML) or isinstance(ds, GaussianMixtureModelFromSK):
			self._loc = ds.components_distribution.loc
			self._cov = ds.components_distribution.covariance()
		else:
			self._cov = ds.covariance()
			self._loc = ds.loc

		self._t = np.linspace(-np.pi, np.pi, nb_segments)

	def plot(self, *args, **kwargs):
		return self.plot_gmm(*args, **kwargs)


	def plot_gmm(self, dim=None, color=[1, 0, 0], alpha=0.5, linewidth=1,
				 markersize=6, batch_idx=0,
				 ax=None, empty=False, edgecolor=None, edgealpha=None,
				 border=False, nb=1, center=True, zorder=20, equal=True, sess=None,
				 feed_dict={}, axis=0):

		if sess is None:
			sess = tf.compat.v1.get_default_session()

		loc, cov = sess.run([self._loc, self._cov], feed_dict)


		if loc.ndim == 3:
			loc = loc[batch_idx] if axis==0 else loc[:, batch_idx]
		if cov.ndim == 4:
			cov = cov[batch_idx] if axis==0 else cov[:, batch_idx]

		if loc.ndim == 1:
			loc = loc[None]
		if cov.ndim == 2:
			cov = cov[None]

		nb_states = loc.shape[0]

		if dim:
			loc = loc[:, dim]
			cov = cov[np.ix_(range(cov.shape[0]), dim, dim)] if isinstance(dim,
																		   list) else cov[
																					  :,
																					  dim,
																					  dim]
		if not isinstance(color, list) and not isinstance(color, np.ndarray):
			color = [color] * nb_states
		elif not isinstance(color[0], str) and not isinstance(color, np.ndarray):
			color = [color] * nb_states

		if not isinstance(alpha, np.ndarray):
			alpha = [alpha] * nb_states
		else:
			alpha = np.clip(alpha, 0.1, 0.9)

		rs = tf.linalg.sqrtm(cov).eval()

		pointss = np.einsum('aij,js->ais', rs, np.array([np.cos(self._t), np.sin(self._t)]))
		pointss += loc[:, :, None]

		for i, c, a in zip(range(0, nb_states, nb), color, alpha):
			points = pointss[i]

			if edgecolor is None:
				edgecolor = c

			polygon = plt.Polygon(points.transpose().tolist(), facecolor=c, alpha=a,
								  linewidth=linewidth, zorder=zorder,
								  edgecolor=edgecolor)

			if edgealpha is not None:
				plt.plot(points[0, :], points[1, :], color=edgecolor)

			if ax:
				ax.add_patch(polygon)  # Patch

				l = None
				if center:
					a = alpha[i]
				else:
					a = 0.

				ax.plot(loc[i, 0], loc[i, 1], '.', color=c, alpha=a)  # Mean

				if border:
					ax.plot(points[0, :], points[1, :], color=c, linewidth=linewidth,
							markersize=markersize)  # Contour
				if equal:
					ax.set_aspect('equal')
			else:
				if empty:
					plt.gca().grid('off')
					# ax[-1].set_xlabel('x position [m]')
					plt.gca().set_axis_bgcolor('w')
					plt.axis('off')

				plt.gca().add_patch(polygon)  # Patch
				l = None

				if center:
					a = alpha[i]
				else:
					a = 0.0

				l, = plt.plot(loc[i, 0], loc[i, 1], '.', color=c, alpha=a)  # Mean
				# plt.plot(points[0,:], points[1,:], color=c, linewidth=linewidth , markersize=markersize) # Contour
				if equal:
					plt.gca().set_aspect('equal')
		return l

def plot_coordinate_system(A, b, scale=1., text=None, equal=True, text_kwargs={},
						   ax=None, **kwargs):
	"""

	:param A:		nb_dim x nb_dim
		Rotation matrix
	:param b: 		nb_dim
		Translation
	:param scale: 	float
		Scaling of the axis
	:param equal: 	bool
		Set matplotlib axis to equal
	:param ax: 		plt.axes()
	:param kwargs:
	:return:
	"""
	a0 = np.vstack([b, b + scale * A[:,0]])
	a1 = np.vstack([b, b + scale * A[:,1]])

	if ax is None:
		p, a = (plt, plt.gca())
	else:
		p, a = (ax, ax)

	if equal and a is not None:
		a.set_aspect('equal')

	p.plot(a0[:, 0], a0[:, 1], 'r', **kwargs)
	p.plot(a1[:, 0], a1[:, 1], 'b', **kwargs)

	if not text is None:
		p.text(b[0]-0.1 * scale, b[1]- 0.15 * scale, text, **text_kwargs)

def plot_coordinate_system_3d(
		A, b, scale=1., equal=True, dim=None, ax=None, text=None, dx_text=[0., 0.],
		text_kwargs={}, **kwargs):
	"""

	:param A:		nb_dim x nb_dim
		Rotation matrix
	:param b: 		nb_dim
		Translation
	:param scale: 	float
		Scaling of the axis
	:param equal: 	bool
		Set matplotlib axis to equal
	:param ax: 		plt.axes()
	:param kwargs:
	:return:
	"""

	if dim is None:
		dim = [0, 1]

	a0 = np.vstack([b[dim], b[dim] + scale * A[dim, 0]])
	a1 = np.vstack([b[dim], b[dim] + scale * A[dim, 1]])
	a2 = np.vstack([b[dim], b[dim] + scale * A[dim, 2]])

	if ax is None:
		p, a = (plt, plt.gca())
	else:
		p, a = (ax, ax)


	if equal and a is not None:
		a.set_aspect('equal')


	if text is not None:
		a.text(b[dim[0]] + dx_text[0], b[dim[1]] + dx_text[1], text, **text_kwargs)

	label = kwargs.pop('label', None)
	color = kwargs.get('color', None)
	if label is not None: kwargs['label'] = label + ' x'
	if color is not None: kwargs['label'] = label

	p.plot(a0[:, 0], a0[:, 1], 'r', **kwargs)

	if color is not None:
		label = kwargs.pop('label', None)

	if label is not None and color is None: kwargs['label'] = label + ' y'
	p.plot(a1[:, 0], a1[:, 1], 'g', **kwargs)
	if label is not None and color is None: kwargs['label'] = label + ' z'
	p.plot(a2[:, 0], a2[:, 1], 'b', **kwargs)
