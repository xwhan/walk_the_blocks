#!/usr/bin/python 

import time
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F

from image_cnn import *
from action_encoder import *
from seq_encoder import * 

class Policy_model(nn.Module):
	"""docstring for Context_attention"""
	def __init__(self, image_embed_dim, hidden_dim, action_dim_1, action_dim_2, inter_dim, attention=False, dis=False):
		super(Policy_model, self).__init__()
		self.image_embed_dim = image_embed_dim
		self.hidden_dim = hidden_dim
		self.block_dim = action_dim_1
		self.direction_dim = action_dim_2
		self.inter_dim = inter_dim
		self.attention = attention
		self.dis = dis

		n_blocks = 20
		n_directions = 4

		self.image_encoder = CNN_encoder(input_channels=3*5, output_size=self.image_embed_dim, image_dim=120)
		self.seq_encoder = Seq_encoder(output_size=self.hidden_dim, embed_dim=150)
		self.action_encoder = Action_encoder(num_directions=n_directions, block_dim=self.block_dim, direction_dim=self.direction_dim)

		self.mlp1 = nn.Linear(self.image_embed_dim + self.hidden_dim + self.direction_dim, self.inter_dim) 
		self.mlp2 = nn.Linear(self.hidden_dim, self.inter_dim) # just predict block id
		self.value_layer = nn.Linear(self.inter_dim, 1)
		self.block_layer = nn.Linear(self.inter_dim, n_blocks)
		self.direction_layer = nn.Linear(self.inter_dim, n_directions + 1)

	def forward(self, inputs):
		""" 
		image: variable of float tensor (1,15,120,120) -> (-1, 15, 120, 120)
		instruction: variable of long tensor -> (-1 * max_lens)
		action: (-1 * 1)
		lens: list of lengths
		"""
		images = inputs[0]
		instructions = inputs[1]
		lens = inputs[2]
		last_actions = inputs[3] # batch_size * 1
		img_embed = self.image_encoder(images) # batch_size * image_embed_dim
		seq_embed = self.seq_encoder(instructions, lens) # max_len * batch_size * hidden

		seq_embed = torch.mean(seq_embed, dim=0) # batch_size * hidden

		action_embed = self.action_encoder(last_actions) # batch_size * 56
		state_embed = self.mlp1(torch.cat((img_embed, seq_embed, action_embed), dim=1)) # batch_size * inter_dim
		state_embed_nlp = self.mlp2(seq_embed)

		direction_prob = F.softmax(self.direction_layer(F.relu(state_embed))) # batch_size * num_actions
		block_prob = F.softmax(self.block_layer(F.relu(state_embed_nlp)))
		values = self.value_layer(F.relu(state_embed))
		return direction_prob, block_prob, values

	def evaluate_action(self, inputs, directions):
		probs, _, _ = self(inputs)
		batch_size = probs.size()[0]
		log_probs = torch.log(probs + 1e-6) # batch * num_actions
		gather_indices = torch.arange(0, batch_size).long().cuda() * 5 + directions
		action_log_probs = log_probs.view(-1)[gather_indices]
		dist_entropy = - (log_probs * probs).sum(-1).mean()
		return action_log_probs, dist_entropy

	def block_loss(self, batch):
		imgs = batch[0]
		instructions = batch[1]
		lens = batch[2]
		last_directions = batch[3]
		gold_blocks = batch[4]

		_, block_probs, _ = self((imgs, instructions, lens, last_directions))
		batch_size = block_probs.size()[0]
		block_gather_indices = torch.arange(0, batch_size).long().cuda() * 20 + gold_blocks
		block_loss = - torch.log(block_probs.view(-1)[block_gather_indices] + 1e-6).mean()
		return block_loss

	def sl_loss(self, batch, entropy_coef):
		imgs = batch[0]
		instructions = batch[1]
		lens = batch[2]
		last_directions = batch[3]
		gold_blocks = batch[4]
		gold_directions = batch[5]

		direction_probs, block_probs, _ = self((imgs, instructions, lens, last_directions))
		direction_log_probs = torch.log(direction_probs + 1e-6)
		dist_entropy = - (direction_log_probs * direction_probs).sum(-1).mean()
		entropy_loss =  - entropy_coef * dist_entropy

		batch_size = direction_probs.size()[0]
		block_gather_indices = torch.arange(0, batch_size).long().cuda() * 20 + gold_blocks
		direction_gather_indices = torch.arange(0, batch_size).long().cuda() * 5 + gold_directions
		block_loss =  - torch.log(block_probs.view(-1)[block_gather_indices] + 1e-6).mean()
		direction_loss = - torch.log(direction_probs.view(-1)[direction_gather_indices] + 1e-6).mean()
		final_loss = block_loss + direction_loss + entropy_loss
		return final_loss, dist_entropy

	def a2c_loss(self, batch, baselines, rewards, args):
		imgs = batch[0]
		instructions = batch[1]
		lens = batch[2]
		last_directions = batch[3]
		gold_blocks = batch[4]
		chosen_directions = batch[5]

		advs = np.array(rewards) - np.array(baselines)
		rewards = Variable(torch.FloatTensor(rewards).cuda())

		direction_probs, block_probs, values = self((imgs, instructions, lens, last_directions))
		direction_entropy = - (torch.log(direction_probs + 1e-6) * direction_probs).sum(-1).mean()	
		entropy_loss = - direction_entropy * args.entropy_coef	

		batch_size = direction_probs.size()[0]
		block_gather_indices = torch.arange(0, batch_size).long().cuda() * 20 + gold_blocks
		block_loss = - torch.log(block_probs.view(-1)[block_gather_indices] + 1e-6).mean()

		direction_gather_indices = torch.arange(0, batch_size).long().cuda() * 5 + chosen_directions
		direction_log_probs = torch.log(direction_probs.view(-1)[direction_gather_indices] + 1e-6)

		adv_targ = Variable(torch.FloatTensor(advs).cuda())

		value_loss = (rewards - values).pow(2).mean()

		action_loss = - (direction_log_probs * adv_targ).mean()

		return (block_loss + entropy_loss + value_loss + action_loss), direction_entropy

	def reinforce_loss(self, batch, rewards, args):
		imgs = batch[0]
		instructions = batch[1]
		lens = batch[2]
		last_directions = batch[3]
		gold_blocks = batch[4]
		chosen_directions = batch[5]

		rewards = Variable(torch.FloatTensor(rewards).cuda())

		direction_probs, block_probs, values = self((imgs, instructions, lens, last_directions))
		direction_entropy = - (torch.log(direction_probs + 1e-6) * direction_probs).sum(-1).mean()	
		entropy_loss = - direction_entropy * args.entropy_coef

		batch_size = direction_probs.size()[0]
		block_gather_indices = torch.arange(0, batch_size).long().cuda() * 20 + gold_blocks
		block_loss = - torch.log(block_probs.view(-1)[block_gather_indices] + 1e-6).mean()

		direction_gather_indices = torch.arange(0, batch_size).long().cuda() * 5 + chosen_directions
		direction_log_probs = torch.log(direction_probs.view(-1)[direction_gather_indices] + 1e-6)

		action_loss = - (direction_log_probs * rewards).mean()

		return	(action_loss + block_loss + entropy_loss), direction_entropy

	def ppo_loss(self, batch, old_model, rewards, baselines, args):
		imgs = batch[0]
		instructions = batch[1]
		lens = batch[2]
		last_directions = batch[3]
		gold_blocks = batch[4]
		chosen_directions = batch[5]

		advs = np.array(rewards) - np.array(baselines)
		rewards = Variable(torch.FloatTensor(rewards).cuda())

		direction_probs, block_probs, values = self((imgs, instructions, lens, last_directions))

		direction_entropy = - (torch.log(direction_probs + 1e-6) * direction_probs).sum(-1).mean()
		entropy_loss = - direction_entropy * args.entropy_coef

		batch_size = direction_probs.size()[0]
		block_gather_indices = torch.arange(0, batch_size).long().cuda() * 20 + gold_blocks
		block_loss = - torch.log(block_probs.view(-1)[block_gather_indices] + 1e-6).mean()

		old_direction_probs, _, _ = old_model((imgs, instructions, lens, last_directions))
		direction_gather_indices = torch.arange(0, batch_size).long().cuda() * 5 + chosen_directions
		direction_log_probs = torch.log(direction_probs.view(-1)[direction_gather_indices] + 1e-6)
		old_direction_log_probs = torch.log(old_direction_probs.view(-1)[direction_gather_indices] + 1e-6)
		ratio = torch.exp(direction_log_probs - Variable(old_direction_log_probs.data))
		adv_targ = Variable(torch.FloatTensor(advs).cuda())
		surr1 = ratio * adv_targ
		surr2 = torch.clamp(ratio, 1.0 - args.clip_epsilon, 1.0 + args.clip_epsilon) * adv_targ
		action_loss = - torch.min(surr1, surr2).mean()
		# action_loss = - (direction_log_probs * adv_targ).mean()

		value_loss = (rewards - values).pow(2).mean()

		return (block_loss + action_loss + value_loss + entropy_loss), direction_entropy

if __name__ == '__main__':
	model = Context_attention(image_embed_dim=200, hidden_dim=200, action_dim_1=32, action_dim_2=24, inter_dim=120)
	image = Variable(torch.randn(1,15,120,120).cuda())
	instruction = Variable(torch.LongTensor(1,15).zero_().cuda())
	action = (Variable(torch.LongTensor([[1]]).cuda()), Variable(torch.LongTensor([[2]]).cuda()))
	start = time.time()
	direction_prob = model(image, instruction, action)
	end = time.time()
	print end - start


