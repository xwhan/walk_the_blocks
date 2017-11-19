#!/usr/bin/python 

import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F

class Action_encoder(nn.Module):
	"""docstring for Action_encoder"""
	def __init__(self, num_directions, block_dim, direction_dim, num_blocks):
		super(Action_encoder, self).__init__()
		self.num_blocks = num_blocks
		self.num_directions = num_directions
		self.direction_dim = direction_dim
		self.block_dim = block_dim

		self.direction_embed = nn.Embedding(self.num_directions + 2, self.direction_dim) # add one direction for no-op, also one for STOP
		self.block_embed = nn.Embedding(self.num_blocks + 1, self.block_dim) # add one for no block

	def forward(self, direction_batch, block_batch):
		"""input dimention
		last_actions: batch_size * 1
		"""
		# direction_batch = last_actions[:,0]
		# block_batch = last_actions[:,1]
		block_embedding = self.block_embed(block_batch)
		direction_embedding = self.direction_embed(direction_batch)
		block_embedding = block_embedding.squeeze(1)
		direction_embedding = direction_embedding.squeeze(1) 
		action_embedding = torch.cat((block_embedding, direction_embedding), dim=1)
		return action_embedding # batch_size * 56
		# return direction_embedding