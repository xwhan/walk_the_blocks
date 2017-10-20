#!/usr/bin/python 

import time
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F

class Agent_with_monitor(nn.Module):
	"""docstring for Agent_with_monitor"""
	def __init__(self, inter_dim, hidden_dim, image_dim, embed_dim):
		super(Agent_with_monitor, self).__init__()
		self.inter_dim = inter_dim
		self.hidden_dim = hidden_dim
		self.image_dim = image_dim
			
		input_channel = 3
		image_size = 12

		self.conv1 = nn.Conv2d(input_channel, 32, kernel_size=3, stride=1, padding=1)
		self.pool1 = nn.MaxPool2d(kernel_size=2) # 60*60*32
		self.conv2 = nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1)
		self.pool2 = nn.MaxPool2d(kernel_size=2) # 30*30*32
		self.conv3 = nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1)
		self.pool3 = nn.MaxPool2d(kernel_size=2) # 15*15*32
		self.conv4 = nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1)
		self.pool4 = nn.MaxPool2d(kernel_size=3) # 5*5*32

		self.affine1 = nn.Linear(5*5*32, inter_dim)

		self.word2id = self.build_wordid()
		vocab_size = len(self.word2id.keys())
		self.rnn = nn.GRU(input_size=embed_dim, hidden_size=hidden_dim)
		self.embed = nn.Embedding(vocab_size, embed_dim)

		# self.attention = nn.Linear(inter_dim, hidden_dim)

		self.final = nn.Linear(inter_dim + hidden_dim, 1)

		self.init_gru_state = Variable(torch.FloatTensor(1, 1, hidden_dim).zero_().cuda())

	def build_wordid(self):
		tokens = open('../BlockWorldSimulator/Assets/vocab_both').readlines()
		word2id = {}

		i = 0
		for token in tokens:
			token = token.rstrip()
			if token not in word2id:
				word2id[token] = i
				i += 1

		word2id["$NULL"] = i
		word2id["$UNK$"] = i+1
		return word2id

	def forward(self, x):
		"""
		x[0]: input_image - 1*3*120*120
		x[1]: input_instruction - 1 * seq_len
		"""
		image_ = F.relu(self.pool1(self.conv1(x[0])))
		image_ = F.relu(self.pool2(self.conv2(image_)))
		image_ = F.relu(self.pool3(self.conv3(image_)))
		image_ = F.relu(self.pool4(self.conv4(image_)))
		image_ = self.affine1(image_.view(-1, 5*5*32)) # 1*(5*5*32)

		seq_embed = self.embed(x[1])
		seq_embed = seq_embed.permute(1,0,2)
		rnn_outputs, _ = self.rnn(seq_embed, self.init_gru_state)
		rnn_outputs = rnn_outputs.squeeze() # seq_len * 2hidden_dim

		seq_embed = torch.mean(rnn_outputs, dim=0, keepdim=True)
		# img_attn = self.attention(image_) # 1* 2hidden
		# img_attn_weight = F.softmax(torch.mm(img_attn, torch.t(rnn_outputs))) # 1 *seq_len
		# seq_embed = torch.t(img_attn_weight) * rnn_outputs
		# seq_embed = torch.sum(seq_embed, dim=0, keepdim=True)

		predict = F.sigmoid(self.final(torch.cat((image_, seq_embed), dim=1)))
		return predict.squeeze()

	def cal_loss(self, inputs, y):
		y_pred = self(inputs)
		return (y_pred - y) ** 2
