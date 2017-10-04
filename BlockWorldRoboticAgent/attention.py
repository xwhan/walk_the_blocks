#!/usr/bin/python 

import time
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F

from image_cnn import *
from action_encoder import *
from seq_encoder import * 

class Context_attention(nn.Module):
	"""docstring for Context_attention"""
	def __init__(self, image_embed_dim, hidden_dim, action_dim_1, action_dim_2, inter_dim):
		super(Context_attention, self).__init__()
		self.image_embed_dim = image_embed_dim
		self.hidden_dim = hidden_dim
		self.block_dim = action_dim_1
		self.direction_dim = action_dim_2
		self.inter_dim = inter_dim

		n_blocks = 20
		n_directions = 4

		self.image_encoder = CNN_encoder(input_channels=3 * 5, output_size=self.image_embed_dim, image_dim=120)
		self.seq_encoder = Seq_encoder(output_size=self.hidden_dim, embed_dim=150)
		self.action_encoder = Action_encoder(num_blocks=n_blocks, num_directions=n_directions, block_dim=self.block_dim, direction_dim=self.direction_dim)

		self.attention_weights = nn.Linear(self.image_embed_dim, self.hidden_dim*2) # potentially add more advanced attention mechanism

		self.mlp1 = nn.Linear(self.image_embed_dim + self.hidden_dim*2 + self.block_dim + self.direction_dim, self.inter_dim)
		self.action_layer = nn.Linear(self.inter_dim, n_blocks*n_directions + 1) # add one for stop

	def forward(self, inputs):
		""" 
		image: tensor variable (1,15,120,120)
		instruction: tensor variable (1,-1)
		action: tensor tuple variable ((1,1),(1,1))
		"""
		image = inputs[0]
		instruction = inputs[1]
		action = inputs[2]
		img_embed = self.image_encoder(image) # 1 * image_embed_dim
		seq_embed = self.seq_encoder(instruction) # seq_len * 1 * 2*hidden
		seq_embed = seq_embed.squeeze(1) # seq_len * 2hidden
		img_attention = self.attention_weights(img_embed) # 1 * (2*hidden)
		img_attention_weights = F.softmax(torch.mm(img_attention, torch.t(seq_embed)))# 1 * seq_len
		seq_embed = torch.t(img_attention_weights) * seq_embed
		seq_embed = torch.sum(seq_embed, dim=0, keepdim=True)

		action_embed = self.action_encoder(action[0], action[1])
		state_embed = self.mlp1(torch.cat((img_embed, seq_embed, action_embed), dim=1))

		action_prob = F.softmax(self.action_layer(F.relu(state_embed)))
		return action_prob

if __name__ == '__main__':
	model = Context_attention(image_embed_dim=200, hidden_dim=200, action_dim_1=32, action_dim_2=24, inter_dim=120)
	image = Variable(torch.randn(1,15,120,120).cuda())
	instruction = Variable(torch.LongTensor(1,15).zero_().cuda())
	action = (Variable(torch.LongTensor([[1]]).cuda()), Variable(torch.LongTensor([[2]]).cuda()))
	start = time.time()
	direction_prob = model(image, instruction, action)
	end = time.time()
	print end - start


