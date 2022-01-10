import numpy as np
import torch as T
import torch.nn as nn
from torch.functional import F

from Lenia import Lenia_C, LeniaStepFFTC
from Spaces import *

class Lenia_Controller(nn.Module):

	@staticmethod
	def default_config() :
		default_config = Dict()
		default_config.version = 'pytorch_fft'  # "pytorch_fft", "pytorch_conv2d"
		default_config.SX = 256
		default_config.SY = 256
		default_config.final_step = 40
		default_config.C = 1
		
		return default_config

	#===================================================================
	def __init__(self, state_channels, hidden_channels, 
		sensory_hidden_kernels = 10, hidden_hidden_kernels = 10, 
		hidden_motor_kernels = 10, device = "cpu", SX = 256,
		SY = 256):

		super().__init__()

		self.state_channels = state_channels
		self.hidden_channels = hidden_channels
		self.sh_kernels = sensory_hidden_kernels
		self.hh_kernels = hidden_hidden_kernels
		self.hm_kernels = hidden_motor_kernels
		self.device = device
		self.SX = SX
		self.SY = SY
		
		self.C = state_channels + hidden_channels + 1

		self.config = self.__class__.default_config()
		self.config.C = self.C

		#Commpute total number of kernels
		# sensory -> hidden
		k = state_channels * hidden_channels * self.sh_kernels
		# hidden -> hidden
		k += (hidden_channels ** 2) * self.hh_kernels
		# hidden -> motor
		k += hidden_channels * self.hm_kernels
		self.total_kernels = k

		self.update_rule_space = LeniaUpdateRuleSpace(self.total_kernels, 
			self.C)

		self.reset()
	#===================================================================
	@property
	def hidden_indexes(self):
		return [i + self.state_channels for i in 
		range(self.hidden_channels)]
		@property
		def sensory_indexes(self):
			return list(range(self.state_channels))
			@property
			def motor_indexes(self):
				return [self.C - 1]
	#===================================================================
	def forward(self):
		self.state = self.lenia_step(self.state)
		return self.state
	#===================================================================
	def encode_state(self, states):
		"""encode current state in sensory channels"""
		assert states.shape == (self.state_channels, self.SX, self.SY)

		for i in range(self.state_channels):
			self.state[0, :, :, i] = states[i]
	#===================================================================
	def reset(self, update_rule_parameters = None):

		if(update_rule_parameters is not None):
			self.update_rule_parameters = update_rule_parameters

		else:
			policy_parameters = Dict.fromkeys(['update_rule'])
			policy_parameters['update_rule'] = self.update_rule_space.sample()
  			#divide h by 3 at the beginning as some unbalanced kernels can easily kill
			policy_parameters['update_rule'].h = policy_parameters['update_rule'].h/3
			self.update_rule_parameters = policy_parameters['update_rule']

        #===================Modify c0 and c1======================
        # 1. sensory -> hidden kernels
		c0 = [0 for _ in range(self.sh_kernels * self.hidden_channels * self.state_channels)]
		c1 = []
		for i in self.hidden_indexes:
			c1 = c1 + [i for _ in range(self.sh_kernels)]

        # 2. hidden -> hidden kernels
		for i in self.hidden_indexes:
			for j in self.hidden_indexes:
				c0 += [i for _ in range(self.hh_kernels)]
				c1 += [j  for _ in range(self.hh_kernels)]

        # 3. hidden -> motor kernels
		c1 += [self.C - 1 for _ in range(self.hidden_channels 
        	* self.hm_kernels)]	
		for i in self.hidden_indexes:
			c0 += [i for _ in range(self.hm_kernels)]

			self.update_rule_parameters["c0"] = T.tensor(c0)
			self.update_rule_parameters["c1"] = T.tensor(c1)

        # initialize Lenia CA with update rule parameters
		if self.config.version == "pytorch_fft":
			lenia_step = LeniaStepFFTC(self.C,
        		self.update_rule_parameters['R'], 
        		self.update_rule_parameters['T'],
        		self.update_rule_parameters['c0'],
        		self.update_rule_parameters['c1'], 
        		self.update_rule_parameters['r'], 
        		self.update_rule_parameters['rk'], 
        		self.update_rule_parameters['b'], 
        		self.update_rule_parameters['w'],
        		self.update_rule_parameters['h'], 
        		self.update_rule_parameters['m'],
        		self.update_rule_parameters['s'],
        		1, is_soft_clip=False, 
        		SX=self.SX, 
        		SY=self.SY, 
        		device=self.device)

		self.add_module('lenia_step', lenia_step)        
		self.to(self.device)
    #===================================================================
	def reset_state(self):
		self.state = T.zeros((1, self.SX, self.SY, self.C)).to(self.device)
	#===================================================================

