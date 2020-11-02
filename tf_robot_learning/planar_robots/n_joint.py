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
from .robot import Robot
pi = 3.14159

class NJointRobot(Robot):
	def __init__(self, n=3, session=None):
		Robot.__init__(self)

		self._ls = tf.constant(1/float(n) * tf.ones(n))

		margin = 0.02

		self._joint_limits = tf.constant([
			[0. + margin, pi - margin]] +
			[[-pi + margin, pi - margin]] * (n - 1), dtype=tf.float32)

		self._dof = n

	def xs(self, q):
		if q.shape.ndims == 1:
			q_currents = tf.cumsum(q)

			x = tf.cumsum(self._ls * tf.cos(q_currents))
			x = tf.concat([tf.zeros(1), x], 0)
			y = tf.cumsum(self._ls * tf.sin(q_currents))
			y = tf.concat([tf.zeros(1), y], 0)

			return tf.transpose(tf.stack([x, y]))


		else:
			q_currents = tf.cumsum(q, axis=1)
			x = tf.cumsum(self._ls[None] * tf.cos(q_currents), axis=1)
			x = tf.concat([tf.zeros_like(x[:, 0][:, None]), x], axis=1)
			y = tf.cumsum(self._ls[None] * tf.sin(q_currents), axis=1)
			y = tf.concat([tf.zeros_like(y[:, 0][:, None]), y], axis=1)

			return tf.concat([x[..., None], y[..., None]], axis=-1)

	def J(self, q):
		if q.shape.ndims == 1:
			X = self.xs(q)[-1]
			dX = tf.reshape(tf.convert_to_tensor([tf.gradients(Xi, q) for Xi in tf.unstack(X)]), (2, self.dof))
			return dX
		else:
			list_dX = []
			for q_single in tf.unstack(q, axis=0):
				X = self.xs(q_single)[-1]
				dX = tf.reshape(tf.convert_to_tensor([tf.gradients(Xi, q_single)
													  for Xi in tf.unstack(X)]), (2, self.dof))
				list_dX.append(dX)
			dX = tf.stack(list_dX)
			return dX


