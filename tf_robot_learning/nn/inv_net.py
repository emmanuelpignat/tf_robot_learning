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

import numpy as np
import tensorflow.compat.v1 as tf
import os
import matplotlib.pyplot as plt
from tensorflow.python.ops import init_ops

class GlorotSmall(init_ops.VarianceScaling):
    """The Glorot uniform initializer, also called Xavier uniform initializer.

    It draws samples from a uniform distribution within [-limit, limit]
    where `limit` is `sqrt(6 / (fan_in + fan_out))`
    where `fan_in` is the number of input units in the weight tensor
    and `fan_out` is the number of output units in the weight tensor.

    Args:
    seed: A Python integer. Used to create random seeds. See
      `tf.set_random_seed`
      for behavior.
    dtype: Default data type, used if no `dtype` argument is provided when
      calling the initializer. Only floating point types are supported.

    References:
      [Glorot et al., 2010](http://proceedings.mlr.press/v9/glorot10a.html)
      ([pdf](http://jmlr.org/proceedings/papers/v9/glorot10a/glorot10a.pdf))
    """

    def __init__(self, scale=1., seed=None, dtype=tf.float32):
        super(GlorotSmall, self).__init__(
            scale=scale,
            mode="fan_avg",
            distribution="uniform",
            seed=seed,
            dtype=dtype)

    def get_config(self):
        return {"seed": self.seed, "dtype": self.dtype.name}


def dense_resnet(inputs, mid_channels, output_channels, num_blocks, activation=None, scale=1.):
    assert len(inputs.shape) == 2
    initializer = GlorotSmall(scale=scale)

    def _blocks(_x, name):
        shortcut = _x
        _x = tf.layers.dense(_x, mid_channels, activation=activation, name=name+'1', kernel_initializer=initializer)
        _x = tf.layers.dense(_x, mid_channels, activation=activation, name=name+'2', kernel_initializer=initializer)
        return _x + shortcut
    
    inputs = tf.layers.dense(inputs, mid_channels, activation=activation    , name='initial', kernel_initializer=initializer)
    
    for i in range(num_blocks):
        inputs = _blocks(inputs, '{}'.format(i))
    inputs = tf.layers.dense(inputs, mid_channels, activation=None, name='final')
    return inputs
        
#Define mask
def get_mask(inputs, reverse_mask, data_format='NHWC', dtype=tf.float32):
    shape = inputs.get_shape().as_list()
    if len(shape) == 2:
        N = shape[-1]
        range_n = tf.range(N)
        odd_ind = tf.mod(range_n, 2)
        
        odd_ind = tf.reshape(odd_ind, [-1, N])
        checker = odd_ind
        
    
    elif len(shape) == 4:
        H = shape[2] if data_format == 'NCHW' else shape[1]
        W = shape[3] if data_format == 'NCHW' else shape[2]
               
        range_h = tf.range(H)
        range_w = tf.range(W)
        
        odd_ind_h = tf.cast(tf.mod(range_h, 2), dtype=tf.bool)
        odd_ind_w = tf.cast(tf.mod(range_w, 2), dtype=tf.bool)
        
        odd_h = tf.tile(tf.expand_dims(odd_ind_h, -1), [1, W])
        odd_w = tf.tile(tf.expand_dims(odd_ind_w,  0), [H, 1])
                
        checker = tf.logical_xor(odd_h, odd_w)
        
        reshape = [-1, 1, H, W] if data_format == 'NCHW' else [-1, H, W, 1]
        checker = tf.reshape(checker, reshape)
        
    
    else:
        raise ValueError('Invalid tensor shape. Dimension of the tensor shape must be '
                         '2 (NxD) or 4 (NxCxHxW or NxHxWxC), got {}.'.format(inputs.get_shape().as_list()))
        
        
    if checker.dtype != dtype:
        checker = tf.cast(checker, dtype)
        
    if reverse_mask:
        checker = 1. - checker
        
    return checker

# Define coupling layer
def coupling_layer(inputs, mid_channels, num_blocks, reverse_mask, activation=None,
                   name='coupling_layer', backward=False, reuse=None, scale=1.):
    mask = get_mask(inputs, reverse_mask)
    with tf.variable_scope(name) as scope:
        if reuse:
            scope.reuse_variables()
            
        if backward:
            v1 = inputs * mask
            v2 = inputs * (1-mask)
            with tf.variable_scope('st1'):
                st1 = dense_resnet(
                    inputs=v1, mid_channels=mid_channels,
                    output_channels=inputs.get_shape().as_list()[1]*2, num_blocks=3,
                    activation=activation, scale=scale
                )
                s1 = st1[:, 0:tf.shape(inputs)[1]]
                rescale1 = tf.get_variable('rescale_s', shape=[inputs.get_shape().as_list()[1]], dtype=tf.float32, initializer=tf.constant_initializer(1.))
                s1 = rescale1 * tf.nn.tanh(s1)
                t1 = st1[:, tf.shape(inputs)[1]:tf.shape(inputs)[1]*2]
                
            u2 = (1-mask)*(v2 - t1)*tf.exp(-s1)
        
            with tf.variable_scope('st2'):
                st2 = dense_resnet(
                    inputs=u2, mid_channels=mid_channels,
                    output_channels=inputs.get_shape().as_list()[1]*2, num_blocks=3,
                    activation=activation, scale=scale
                )
                s2 = st2[:, 0:tf.shape(inputs)[1]]
                rescale2 = tf.get_variable('rescale_s', shape=[inputs.get_shape().as_list()[1]], dtype=tf.float32, initializer=tf.constant_initializer(1.))
                s2 = rescale2 * tf.nn.tanh(s2)
                t2 = st2[:, tf.shape(inputs)[1]:tf.shape(inputs)[1]*2]
                
            u1 = mask * (v1 - t2)*tf.exp(-s2)
            inputs = u1 + u2
        
        else:
            u1 = inputs * mask
            u2 = inputs * (1-mask)
        
            with tf.variable_scope('st2'):
                st2 = dense_resnet(
                    inputs=u2, mid_channels=mid_channels,
                    output_channels=inputs.get_shape().as_list()[1]*2, num_blocks=3,
                    activation=activation, scale=scale
                )
                s2 = st2[:, 0:tf.shape(inputs)[1]]
                rescale2 = tf.get_variable('rescale_s', shape=[inputs.get_shape().as_list()[1]], dtype=tf.float32, initializer=tf.constant_initializer(1.))
                s2 = rescale2 * tf.nn.tanh(s2)
                t2 = st2[:, tf.shape(inputs)[1]:tf.shape(inputs)[1]*2]
        
            v1 = mask * (u1 * tf.exp(s2) + t2)
            
            with tf.variable_scope('st1'):
                st1 = dense_resnet(
                    inputs=v1, mid_channels=mid_channels,
                    output_channels=inputs.get_shape().as_list()[1]*2, num_blocks=3,
                    activation=activation,scale=scale
                )
                s1 = st1[:, 0:tf.shape(inputs)[1]]
                rescale1 = tf.get_variable('rescale_s', shape=[inputs.get_shape().as_list()[1]], dtype=tf.float32, initializer=tf.constant_initializer(1.))
                s1 = rescale1 * tf.nn.tanh(s1)
                t1 = st1[:, tf.shape(inputs)[1]:tf.shape(inputs)[1]*2]
        
        
            v2 = (1-mask) * (u2 * tf.exp(s1) + t1)
            inputs = v1 + v2
        
        return inputs
    
# Code from https://github.com/chrischute/real-nvp
def preprocess(x):
    data_constraint = 0.9
    y = (x*255. + tf.random.uniform(tf.shape(x), 0, 1))/256.
    y = (2 * y - 1) * data_constraint
    y = (y + 1) / 2
    y = tf.log(y) - tf.log(1-y)
    
    ldj = tf.nn.softplus(y) + tf.nn.softplus(-y) - tf.nn.softplus(tf.log(1-data_constraint) - tf.log(data_constraint))
    sldj = tf.reduce_sum(tf.reshape(ldj, [tf.shape(ldj)[0], -1]), axis=-1)
    return y, sldj


def get_nvp_trainable_variables():
    variables = []

    for i in range(1, 5):
        variables += tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='c%d' % i)

    return variables

def real_nvp(inputs, mid_channels, backward=False, reuse=False,
             activation=tf.nn.tanh, scale_kernel=1., name=''):
#    
    x = inputs
    if backward:
       
        x = coupling_layer(x, mid_channels, 4, activation=activation,
                           reverse_mask=True, name=name + 'c4', backward=backward, reuse=reuse,
                           scale=scale_kernel)
        x = coupling_layer(x, mid_channels, 4, activation=activation,
                           reverse_mask=False, name= name + 'c3', backward=backward, reuse=reuse,
                           scale=scale_kernel)
        x = coupling_layer(x, mid_channels, 4, activation=activation,
                           reverse_mask=True, name=name + 'c2', backward=backward, reuse=reuse,
                           scale=scale_kernel)
        x = coupling_layer(x, mid_channels, 4, activation=activation,
                           reverse_mask=False, name=name +'c1', backward=backward, reuse=reuse,
                           scale=scale_kernel)
    else:
        
#        x, sldj = preprocess(inputs)
        x = coupling_layer(x, mid_channels, 4, activation=activation,
                           reverse_mask=False, name=name + 'c1', backward=backward, reuse=reuse,
                           scale=scale_kernel)
        x = coupling_layer(x, mid_channels, 4, activation=activation,
                           reverse_mask=True, name=name + 'c2', backward=backward, reuse=reuse,
                           scale=scale_kernel)
        x = coupling_layer(x, mid_channels, 4, activation=activation,
                           reverse_mask=False, name=name + 'c3', backward=backward, reuse=reuse,
                           scale=scale_kernel)
        x = coupling_layer(x, mid_channels, 4, activation=activation,
                           reverse_mask=True, name=name + 'c4', backward=backward, reuse=reuse,
                           scale=scale_kernel)
#    x = tf.nn.sigmoid(x)
    return x