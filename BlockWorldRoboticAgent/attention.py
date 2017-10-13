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
	def __init__(self, image_embed_dim, hidden_dim, action_dim_1, action_dim_2, inter_dim, attention=False, dis=False):
		super(Context_attention, self).__init__()
		self.image_embed_dim = image_embed_dim
		self.hidden_dim = hidden_dim
		self.block_dim = action_dim_1
		self.direction_dim = action_dim_2
		self.inter_dim = inter_dim
		self.attention = attention
		self.dis = dis

		n_blocks = 20
		n_directions = 4

		self.image_encoder = CNN_encoder(input_channels=3 * 5, output_size=self.image_embed_dim, image_dim=120)
		self.seq_encoder = Seq_encoder(output_size=self.hidden_dim, embed_dim=150)
		self.action_encoder = Action_encoder(num_blocks=n_blocks, num_directions=n_directions, block_dim=self.block_dim, direction_dim=self.direction_dim)

		if attention:	
			self.attention_weights = nn.Linear(self.image_embed_dim, self.hidden_dim*2) # potentially add more advanced attention mechanism

		self.mlp1 = nn.Linear(self.image_embed_dim + self.hidden_dim + self.block_dim + self.direction_dim, self.inter_dim)
		if dis:
			self.value_layer = nn.Linear(self.inter_dim, n_blocks*n_directions + 1)
		else:
			self.action_layer = nn.Linear(self.inter_dim, n_blocks*n_directions + 1) # add one for stop

	def forward(self, inputs):
		""" 
		image: tensor variable (1,15,120,120) -> (-1, 15, 120, 120)
		instruction: tensor variable (1,-1) -> (-1 * max_lens)
		action: tensor tuple variable ((1,1),(1,1))  -> (-1 * 2)
		lens: (-1, 1)
		"""
		images = inputs[0]
		instructions = inputs[1]
		lens = inputs[2]
		last_actions = inputs[3]
		img_embed = self.image_encoder(images) # batch_size * image_embed_dim
		seq_embed = self.seq_encoder(instructions, lens) # max_len * batch_size * hidden

		if self.attention:
			img_attention = self.attention_weights(img_embed) # 1 * (2*hidden)
			img_attention_weights = F.softmax(torch.mm(img_attention, torch.t(seq_embed)))# 1 * seq_len
			seq_embed = torch.t(img_attention_weights) * seq_embed
			seq_embed = torch.sum(seq_embed, dim=0, keepdim=True)
		else:
			seq_embed = torch.mean(seq_embed, dim=0, keepdim=True) # batch_size * hidden

		action_embed = self.action_encoder(last_actions) # batch_size * 56
		state_embed = self.mlp1(torch.cat((img_embed, seq_embed, action_embed), dim=1)) # batch_size * inter_dim

		if self.dis:
			D_values = F.sigmoid(self.value_layer(F.relu(state_embed))) # batch_size * num_actions
			return D_values
		else:
			action_prob = F.softmax(self.action_layer(F.relu(state_embed))) # batch_size * num_actions
			return action_prob

	def evaluate_action(self, inputs, actions):
		probs = self(inputs)
		batch_size = probs.size()[0]
		log_probs = torch.log(probs + 1e-13) # batch * num_actions
		gather_indices = torch.arange(0, batch_size)*batch_size + actions
		action_log_probs = log_probs.view(-1).index_select(gather_indices)
		dist_entropy = - (log_probs * probs).sum(-1).mean()
		return action_log_probs, dist_entropy

if __name__ == '__main__':
	model = Context_attention(image_embed_dim=200, hidden_dim=200, action_dim_1=32, action_dim_2=24, inter_dim=120)
	image = Variable(torch.randn(1,15,120,120).cuda())
	instruction = Variable(torch.LongTensor(1,15).zero_().cuda())
	action = (Variable(torch.LongTensor([[1]]).cuda()), Variable(torch.LongTensor([[2]]).cuda()))
	start = time.time()
	direction_prob = model(image, instruction, action)
	end = time.time()
	print end - start


