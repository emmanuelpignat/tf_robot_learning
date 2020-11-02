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

from .frame import Frame
from .joint import Joint
from .utils import *

class Segment(object):
	def __init__(self, joint, f_tip, child_name='', link=None, fixed=True):
		"""
		Segment of a kinematic chain

		:param joint:
		:type joint: tk.Joint
		:param f_tip:
		:type f_tip: tk.Frame
		"""
		self.joint = joint
		self.f_tip = joint.pose(0.).inv() *  f_tip

		if fixed:
			self.f_tip = self.f_tip.fix_it()

		# print (self.f_tip)
		# print (self.f_tip.fix_it())
		self.child_name = child_name

		self.link = link

		self.pose_0 = self.pose(0.).fix_it()

	def pose(self, q):
		return self.joint.pose(q) * self.f_tip

	def twist(self, q, qdot=0.):
		return self.joint.twist(qdot).ref_point(
			matvecmul(self.joint.pose(q).m, self.f_tip.p))

