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

		self.reset_memory()
	#===================================================================
	def choose_action(self, state):
		self.encode_state(state)

		for _ in range(self.forward_steps):
			self.controller.forward()

		state = self.controller.state

		motor_state = state[..., self.controller.C - 1]
	#===================================================================
	def encode_state(self, state):
		"""encodes current state in lenia controller sensory channels"""
		pass
	#===================================================================
	def learn(self):
		pass
	#===================================================================
	def reset_memory(self):
        self.states_mem = []
        self.rews_mem = []
        self.action_mem = []
        self.episode_indexes = []
    #===================================================================
    def compute_actions_kernels(self):
    	self.action_kernels = None