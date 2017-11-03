#!/usr/bin/python 

import time
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
import numpy as np

from image_cnn import *
from action_encoder import *
from seq_encoder import * 

class Critic_model(nn.Module):
	"""docstring for Critic_model"""
	def __init__(self, image_embed_dim, hidden_dim, direction_dim, inter_dim):
		super(Critic_model, self).__init__()
		self.image_embed_dim = image_embed_dim
		self.hidden_dim = hidden_dim
		self.direction_dim = direction_dim
		self.inter_dim = inter_dim

		n_directions = 4

		self.image_encoder = CNN_encoder(input_channels=3*5, output_size=self.image_embed_dim, image_dim=120)
		self.seq_encoder = Seq_encoder(output_size=self.hidden_dim, embed_dim=150)
		self.action_encoder = Action_encoder(num_directions=n_directions, block_dim=0, direction_dim=self.direction_dim)

		self.mlp = nn.Linear(self.image_embed_dim + self.hidden_dim + self.direction_dim, self.inter_dim)	
		self.critic_layer = nn.Linear(self.inter_dim, n_directions + 1)

	def forward(self, inputs):
		images = inputs[0]
		instructions = inputs[1]
		lens = inputs[2]
		last_actions = inputs[3] # batch_size * 1
		img_embed = self.image_encoder(images) # batch_size * image_embed_dim
		seq_embed = self.seq_encoder(instructions, lens) # max_len * batch_size * hidden

		seq_embed = torch.mean(seq_embed, dim=0) # batch_size * hidden

		action_embed = self.action_encoder(last_actions) # batch_size * 56
		state_embed = self.mlp(torch.cat((img_embed, seq_embed, action_embed), dim=1)) # batch_size * inter_dim
		critic_values = F.sigmoid(self.critic_layer(F.relu(state_embed)))
		return critic_values

	def sample_from_values(self, values):
		values = values.squeeze()
		values = values.data.cpu().numpy()
		ps = np.exp(values)
		action_prob = ps / np.sum(ps)
		direction_id = np.random.choice(np.arange(len(action_prob)), p = action_prob)
		return direction_id

	def inverse_loss(self, random_memory, expert_mempry):
		random_batch = self.build_inputs_memory(random_memory)
		values_random = self((random_batch[0], random_batch[1], random_batch[2], random_batch[3]))
		batch_size_1 = values_random.size()[0]
		random_direction_gather_indices = torch.arange(0, batch_size_1).long().cuda() * 5 + random_batch[4]
		random_loss = - torch.log(1 - values_random.view(-1)[random_direction_gather_indices] + 1e-6).mean()

		expert_batch = self.build_inputs_memory(expert_mempry)
		values_expert = self((expert_batch[0], expert_batch[1], expert_batch[2], expert_batch[3]))
		batch_size_2 = values_expert.size()[0]
		expert_direction_gather_indices = torch.arange(0,batch_size_2).long().cuda() * 5 + expert_batch[4]
		expert_loss = - torch.log(values_expert.view(-1)[expert_direction_gather_indices] + 1e-6).mean()

		return expert_loss + random_loss


	def build_inputs_memory(self, memory, max_instruction=83):
		image_batch = []
		instruction_batch = []
		lens_batch = []
		previous_batch = []
		direction_batch = []

		for exp in memory:
			direction_batch.append(exp[1])
			state = exp[0]
			imgs = np.concatenate(list(state[0]), axis=0)
			imgs = np.expand_dims(imgs, axis=0)
			image_batch.append(imgs)
			instruction_id = state[1]
			lens_batch.append(len(instruction_id))
			instruction_id_padded = np.lib.pad(instruction_id, (0, max_instruction - len(instruction_id)), 'constant', constant_values=(0,0))
			instruction_id_padded = np.expand_dims(instruction_id_padded, axis=0)
			instruction_batch.append(instruction_id_padded)
			previous_direction = state[2]
			previous_batch.append(previous_direction)

		image_batch = Variable(torch.from_numpy(np.concatenate(image_batch, axis=0)).float().cuda())
		instruction_batch = Variable(torch.LongTensor(np.concatenate(instruction_batch, axis=0)).cuda())
		previous_batch = Variable(torch.LongTensor(previous_batch).cuda())
		direction_batch = torch.LongTensor(direction_batch).cuda()

		return	(image_batch, instruction_batch, lens_batch, previous_batch, direction_batch)











