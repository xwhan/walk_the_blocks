#!/usr/bin/python 

import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F

class Action_encoder(nn.Module):
	"""docstring for Action_encoder"""
	def __init__(self, num_blocks, num_directions, block_dim, direction_dim):
		super(Action_encoder, self).__init__()
		self.num_blocks = num_blocks
		self.num_directions = num_directions
		self.block_dim = block_dim
		self.direction_dim = direction_dim

		self.block_embed = nn.Embedding(self.num_blocks + 1, self.block_dim)
		self.direction_embed = nn.Embedding(self.num_directions + 2, self.direction_dim) # add one direction for no-op

	def forward(self, block, direction):
		"""input dimention
		block: 1*1 
		direction: 1*1
		"""
		block_embedding = self.block_embed(block)
		direction_embedding = self.direction_embed(direction)
		block_embedding = block_embedding.squeeze(0)
		direction_embedding = direction_embedding.squeeze(0) 
		action_embedding = torch.cat((block_embedding, direction_embedding), dim=1)
		return action_embedding