#!/usr/bin/python 

import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
import numpy as np
import nltk

class Seq_encoder(nn.Module):
	"""docstring for Seq_encoder"""
	def __init__(self, output_size, embed_dim, pretrain=False):
		super(Seq_encoder, self).__init__()
		self.output_size = output_size
		self.embed_dim = embed_dim
		self.null = "$NULL"
		self.unk = "$UNK$"
		
		self.word2id = self.build_wordid()
		vocab_size = len(self.word2id.keys())
		self.embed_M = nn.Embedding(vocab_size, self.embed_dim)

		self.lstm = nn.LSTM(input_size=self.embed_dim, hidden_size=self.output_size, num_layers=1)

		# self.init_lstm_state = (Variable(torch.FloatTensor(1, 1, self.output_size).zero_().cuda()), Variable(torch.FloatTensor(1, 1, self.output_size).zero_().cuda()))

	def forward(self, x, lens):
		"""input: a sequence of word indice (batch_size * num_of_indices)"""
		x = self.embed_M(x) # batch_size * max_len * embed_dim 
		x = x.permute(1,0,2)
		batch_size = x.size()[1]
		max_len = x.size()[0]
		init_lstm_state = (Variable(torch.FloatTensor(1, batch_size, self.output_size).zero_().cuda()), Variable(torch.FloatTensor(1, batch_size, self.output_size).zero_().cuda()))
		rnn_outputs, _ = self.lstm(x, init_lstm_state)
		mask = np.zeros((max_len, batch_size))
		for i in range(batch_size):
			mask[:lens[i],i] = 1
		mask = Variable(torch.from_numpy(mask).cuda().float()).unsqueeze(2)
		mask = mask.expand_as(rnn_outputs)

		run_outputs = mask * rnn_outputs
		return rnn_outputs # max_len * batch * hidden

	def build_wordid(self):
		tokens = open('../../BlockWorldSimulator/Assets/vocab_both').readlines()
		word2id = {}

		i = 0
		for token in tokens:
			token = token.rstrip()
			if token not in word2id:
				word2id[token] = i
				i += 1

		word2id[self.null] = i
		word2id[self.unk] = i+1
		return word2id

	def instruction2id(self, s):
		"""input: a string of instructions"""
		indices = []
		token_s = nltk.word_tokenize(s)
		for word in token_s:
			word = word.lower()
			if word in self.word2id:
				indices.append(self.word2id[word])
			else:
				indices.append(self.word2id[self.unk])
		return indices




