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

from .import_pykdl import *
import numpy as np

# @profile
def frame_to_np(frame, layout=None, vec=False):
	if vec:
		return np.array([
					frame.p[0], frame.p[1], frame.p[2],
					frame.M[0, 0], frame.M[1, 0], frame.M[2, 0],
					frame.M[0, 1], frame.M[1, 1], frame.M[2, 1],
					frame.M[0, 2], frame.M[1, 2], frame.M[2, 2],
				])
	else:
		return np.array([
			frame.p[0], frame.p[1], frame.p[2],
			frame.M[0, 0], frame.M[0, 1], frame.M[0, 2],
			frame.M[1, 0], frame.M[1, 1], frame.M[1, 2],
			frame.M[2, 0], frame.M[2, 1], frame.M[2, 2],
		])

def forward_kinematic(q, chain):
	nb_jnt = len(q) if isinstance(q, list) else q.shape[0]
	kdl_array = kdl.JntArray(nb_jnt)

	for j in range(nb_jnt):
		kdl_array[j] = q[j]

	end_frame = kdl.Frame()
	solver = kdl.ChainFkSolverPos_recursive(chain)

	solver.JntToCart(kdl_array, end_frame)

	return frame_to_np(end_frame)

class FKSolver(object):
	def __init__(self, chain, nb_jnt):
		"""

		:param chain:
		:param nb_jnt:	Number of joints
		:type nb_jnt:
		"""
		self.nb_jnt = nb_jnt
		self.chain = chain
		self.kdl_array = kdl.JntArray(nb_jnt)

		self.end_frame = kdl.Frame()
		self.solver = kdl.ChainFkSolverPos_recursive(chain)

	# @profile
	def solve(self, q, vec=False):
		for j in range(self.nb_jnt):
			self.kdl_array[j] = q[j]

		self.solver.JntToCart(self.kdl_array, self.end_frame)

		return frame_to_np(self.end_frame, vec=vec)


class JacSolver(object):
	def __init__(self, chain, nb_jnt):
		self.nb_jnt = nb_jnt
		self.chain = chain
		self.kdl_array = PyKDL.JntArray(nb_jnt)

		self.end_jacobian = PyKDL.Jacobian()
		self.solver = PyKDL.ChainJntToJacSolver(chain)

	# @profile
	def solve(self, q):
		for j in range(self.nb_jnt):
			self.kdl_array[j] = q[j]

		self.solver.JntToJac(self.kdl_array, self.end_frame)

		return frame_to_np(self.end_frame)


