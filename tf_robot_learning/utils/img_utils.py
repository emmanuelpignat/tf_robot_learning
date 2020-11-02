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
from .tf_utils import log_normalize

def spatial_soft_argmax(img_array, temp=1.):
	"""
	Compute spatial soft argmax

	:param img_array: [nb_samples, height, width]
	:type img_array: tf.Tensor
	:return:
	"""

	if len(img_array.get_shape()) == 3:
		height = img_array.get_shape()[-2].value
		width = img_array.get_shape()[-1].value
	else:
		height = img_array.get_shape()[-3].value
		width = img_array.get_shape()[-2].value
		n_channel = img_array.get_shape()[-1].value

	img_coords = tf.constant(
		np.array([np.resize(np.arange(height), (height, width)),
				  np.resize(np.arange(width), (width, height)).T]).astype(
			np.float32))

	# softmax = tf.nn.relu(vec_img)/tf.reduce_sum(tf.nn.relu(vec_img))
	if len(img_array.get_shape()) == 3:
		# tf.reduce_sum(
		# 	img_coords[None, :, :, :, None] * tf.exp(
		# 		log_normalize(temp * img_array, axis=(0, 1)))[:, None], axis=(2, 3))
		#
		vec_img = tf.reshape(img_array, [-1, height * width])
		softmax = tf.nn.softmax(vec_img)
		softmax = tf.reshape(softmax, [-1, height, width])
		return tf.einsum('aij,bij->ab', softmax, img_coords)
	else:
		softmax = tf.exp(log_normalize(temp * img_array, axis=(1, 2)))
		return tf.reduce_sum(
			img_coords[None, :, :, :, None] *softmax[:, None], axis=(2, 3)), softmax


def spatial_soft_argmax_temporal(img_array, prev_argmax, temp=1., dist=0.2):
	"""
	Compute spatial soft argmax

	:param img_array: [nb_samples, height, width]
	:type img_array: tf.Tensor
	:return:
	"""

	height = img_array.get_shape()[-3].value
	width = img_array.get_shape()[-2].value

	img_coords = tf.constant(
		np.array([np.resize(np.arange(height), (height, width)),
				  np.resize(np.arange(width), (width, height)).T]).astype(
			np.float32))

	temp_log_prob = tf.reduce_sum(
		((prev_argmax[:, :, None, None] - img_coords[None, :, :, :, None]) ** 2)/(height * width * dist), axis=(1))

	softmax = tf.exp(log_normalize(temp * img_array - temp_log_prob, axis=(1, 2)))
	return tf.reduce_sum(
		img_coords[None, :, :, :, None] *softmax[:, None], axis=(2, 3)), softmax