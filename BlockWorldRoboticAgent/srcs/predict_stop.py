#!/usr/bin/python 

import time
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
import argparse
from tqdm import tqdm
import pickle
import numpy as np
import random
import constants

from tensorboard_logger import configure, log_value
from agent_with_monitor import *

def inputs_from_exp(exp):
	state = exp[0]
	images = state[0]
	instruction_ids = state[1]
	action = exp[1]
	current_image = images[-1]
	if action == 80:
		y = 1
	else:
		y = 0
	image_tensor = Variable(torch.from_numpy(current_image).float().cuda().unsqueeze(0)) # 1*3*120*120
	instruction_input = Variable(torch.LongTensor(instruction_ids).cuda().unsqueeze(0)) # 1*seq_len
	return (image_tensor, instruction_input), y

def predict_stop(agent):
	parser = argparse.ArgumentParser(description='Goal Monitor hyperparameters')
	parser.add_argument('-max_epochs', type=int, default=1)
	parser.add_argument('-lr', type=float, default=0.001)
	parser.add_argument('-clip_value', type=float, default=5.0)
	parser.add_argument('-batch_size', type=int, default=32)
	args = parser.parse_args()

	constants_hyperparam = constants.constants
	train_size = constants_hyperparam["train_size"]		

	optimizer = torch.optim.Adam(agent.parameters())

	configure('runs/add_conv_learn_to_predict_stop_no_biGRU_attn', flush_secs=0.5)

	num_batches = (train_size - 1) / args.batch_size + 1
	step = 0

	train_size = 500

	for epoch in range(args.max_epochs):
		f = open('demonstrations.pkl', 'rb')
		print 'Training Epoch %d' % epoch
		epoch_loss = 0
		batch_loss = 0.0
		for _ in range(train_size):
			batch_count = 0
			memory = pickle.load(f)
			pos_sample = memory[-1]
			neg_samples = memory[:3]
			samples = neg_samples + [pos_sample]
			for exp in samples:
				inputs, label = inputs_from_exp(exp)
				batch_loss += agent.cal_loss(inputs, label)
				batch_count += 1
				epoch_loss += batch_loss.data.cpu().numpy()

			if batch_count == args.batch_size or (_ == train_size -1):
				batch_count = 0
				agent.zero_grad()
				batch_loss.backward()
				# torch.nn.utils.clip_grad_norm(agent.parameters(), 5.0)
				optimizer.step()
				batch_loss = 0.0

		log_value('epoch_loss', epoch_loss, epoch)

	save_path = 'models/stop_model_first2'
	torch.save(agent.state_dict(), save_path)
	print 'model saved'

def test(agent, save_path):
	agent.load_state_dict(torch.load(save_path))
	f = open('demonstrations.pkl', 'rb')

	test_size = 5
	for _ in range(test_size):
		memory = pickle.load(f)
		predicts = []
		gts = []
		for exp in memory:
			inputs, label = inputs_from_exp(exp)
			gts.append(label)
			label_pred = agent(inputs).data.cpu().numpy()
			predicts.append(label_pred)
		print predicts
		print gts

if __name__ == '__main__':
	agent = Agent_with_monitor(256, 64, 120, 25)
	agent.cuda()
	predict_stop(agent)
	# test(agent, 'models/stop_model')


