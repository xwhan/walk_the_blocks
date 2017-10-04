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
		if pretrain:
			pass

		self.bi_lstm = nn.LSTM(input_size=self.embed_dim, hidden_size=self.output_size, num_layers=1, bidirectional=True)

		self.init_lstm_state = (Variable(torch.FloatTensor(1*2, 1, self.output_size).zero_().cuda()), Variable(torch.FloatTensor(1*2, 1, self.output_size).zero_().cuda()))

	def forward(self, x):
		"""input: a sequence of word indice (batch_size * num_of_indices)"""
		x = self.embed_M(x) # batch_size * num_of_indices * embed_dim 
		x = x.unsqueeze(1)
		rnn_outputs, _ = self.bi_lstm(x, self.init_lstm_state)
		return rnn_outputs # seq_len * 1 * hidden*2

	def build_wordid(self):
		tokens = open('../BlockWorldSimulator/Assets/vocab_both').readlines()
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




