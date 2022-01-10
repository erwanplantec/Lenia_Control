"""
Abstract class for an agent controlled with a Lenia controller
In order to be used, inheriting class must override encode_state, decode_action 
and learn methods : 
	- encode_state must map environment states/observations to controller's 
	input channels
	- decode_action maps motor channels to policy distribution over actions
	- learn defines learniong algorithm
"""


import numpy as np
import torch as T
import torch.nn as nn
from torch.functional import F

from Controller import Lenia_Controller

class Lenia_Agent :
	#===================================================================
	def __init__(self, state_dims, n_actions, state_channels, hidden_channels,
		state_hidden_kernels, hidden_hidden_kernels, hidden_motor_kernels,
		forward_steps = 10, SX = 256, SY = 256, device  = "cpu"):

		self.state_dims = state_dims
		self.n_actions = n_actions
		
		self.controller = Lenia_Controller(
				state_channels,
				hidden_channels,
				state_hidden_kernels,
				hidden_hidden_kernels,
				hidden_motor_kernels,
				device, SX, SY
			)
		self.forward_steps = forward_steps

		self.optimizer = T.optim.Adam(self.controller.parameters())
	#===================================================================
	def choose_action(self, state):
		self.encode_state(state)

		for _ in range(self.forward_steps):
			self.controller.forward()

		state = self.controller.state

		motor_state = state[..., self.controller.C - 1]

		a_dist = self.decode_action(motor_state)

		a = a_dist.sample()

		return a.detach().numpy()
	#===================================================================
	def encode_state(self, state):
		"""encodes current state in lenia controller sensory channels"""
		raise NotImplementedError
	#===================================================================
	def decode_action(self, m_state):
		"""return action distribution based on motor channel state m_state"""
		raise NotImplementedError
	#===================================================================
	def learn(self):
		raise NotImplementedError
