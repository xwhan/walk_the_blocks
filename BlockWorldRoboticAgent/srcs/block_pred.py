#!/usr/bin/python 

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from irl_agent import *
from recordtype import recordtype
import time
from tqdm import tqdm
import pickle
import constants
from config import Config
import argparse
from copy import deepcopy

from tensorboard_logger import configure, log_value


def predict_block(agent):
	parser = argparse.ArgumentParser(description='Supervised Training hyperparameters')
	parser.add_argument('-max_epochs', type=int, default=1, help='training epochs')
	parser.add_argument('-batch_size', type=int, default=32, help='batch size for demonstrations')
	parser.add_argument('-lr', type=float, default=0.005, help='learning rate')
	parser.add_argument('-replay_memory_size', type=int, default=32000, help='random shuffle')
	args = parser.parse_args()
	parameters = agent.policy_model.parameters()
	optimiter = torch.optim.Adam(parameters, lr=args.lr)

	num_experiences = 184131
	for epoch in range(args.max_epochs):
		f = open('demonstrations.pkl', 'rb')
		replay_memory = []
		replay_memory_size = args.replay_memory_size
		exp_used = 0
		memory = []
		while exp_used < num_experiences:
			if num_experiences - exp_used < replay_memory_size:
				replay_memory_size = num_experiences - exp_used
			while len(replay_memory) < replay_memory_size:
				# print 'refill the replay memory'
				if len(memory) == 0:
					memory = pickle.load(f)
				exp = memory.pop(0)
				replay_memory.append(exp)
			np.random.shuffle(replay_memory)
			num_batches = (len(replay_memory) - 1) / args.batch_size + 1
			for batch_index in range(num_batches):
				loss = 0.0
				if batch_index == num_batches - 1:
					end = len(replay_memory)
				else:
					end = args.batch_size * (batch_index + 1)
				count = 0
				for exp in replay_memory[(args.batch_size*batch_index):end]:
					state = exp[0]
					action_id = exp[1]
					last_direction = state[2][0]
					state_ = (state[0], state[1], last_direction)
					gold_block = action_id / 4
					inputs = agent.build_batch_inputs([(state_, 0)])
					_, block_prob = agent.policy_model(inputs)
					block_prob = block_prob.squeeze()
					if action_id!=80:
						loss += - torch.log(block_prob[gold_block] + 1e-8)
						count += 1
				loss = loss / count
				optimiter.zero_grad()
				loss.backward()
				optimiter.step()

			replay_memory = []
			exp_used += replay_memory_size

	savepath = 'models/block_pred.pth'
	torch.save(agent.policy_model.state_dict(), savepath)
	print 'Model saved'

def test(agent, mode='dev'):
	agent.policy_model.load_state_dict(torch.load('models/block_pred.pth'))

	config = Config.parse("../BlockWorldSimulator/Assets/config.txt")
	constants_hyperparam = constants.constants
	if mode == 'dev':
		assert config.data_mode == Config.DEV
		test_size = constants_hyperparam["dev_size"]
	elif mode == 'test':
		assert config.data_mode == Config.TEST 
		test_size = constants_hyperparam['test_size']
	else:
		assert config.data_mode == Config.TRAIN 
		test_size = 1000

	block_right = 0.0
	for sample_id in tqdm(range(test_size)):
		img_state = collections.deque([], 5)
		init_imgs = agent.policy_model.image_encoder.build_init_images()
		for img in init_imgs:
			img_state.append(img)
		(status_code, bisk_metric, img, instruction, trajectory) = agent.receive_instruction_image()
		gold_block_id = trajectory[0] / 4
		img = np.transpose(img, (2,0,1))
		img_state.append(img)
		previous_direction = agent.null_previous_direction
		instruction_ids = agent.policy_model.seq_encoder.instruction2id(instruction)
		state = (img_state, instruction_ids, previous_direction)
		inputs = agent.build_batch_inputs([(state, 0)])
		_, block_prob = agent.policy_model(inputs)
		gold_block_id = trajectory[0] / 4
		block_pred = agent.sample_policy(block_prob, method='greedy')
		if block_pred == gold_block_id:
			block_right += 1
		agent.connection.send_message('Ok-Reset')

	print block_right / test_size

if __name__ == '__main__':
	agent = Inverse_agent()
	agent.policy_model.cuda()
	predict_block(agent)
	# test(agent)
