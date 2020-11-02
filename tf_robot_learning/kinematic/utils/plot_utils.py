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
import numpy as np

def axis_equal_3d(ax):
	extents = np.array([getattr(ax, 'get_{}lim'.format(dim))() for dim in 'xyz'])
	sz = extents[:,1] - extents[:,0]
	centers = np.mean(extents, axis=1)
	maxsize = max(abs(sz))
	r = maxsize/2
	for ctr, dim in zip(centers, 'xyz'):
		getattr(ax, 'set_{}lim'.format(dim))(ctr - r, ctr + r)

def plot_robot(xs, color='k', xlim=None,ylim=None, **kwargs):
	dims = kwargs.pop('dims', [0, 1])

	l = plt.plot(xs[:, dims[0]], xs[:,dims[1]], marker='o', color=color, lw=10, mfc='w', solid_capstyle='round',
			 **kwargs)

	plt.axes().set_aspect('equal')

	if xlim is not None: plt.xlim(xlim)
	if ylim is not None: plt.ylim(ylim)

	return l