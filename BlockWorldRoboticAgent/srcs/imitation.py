#!/usr/bin/python 

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from agent import *
from recordtype import recordtype
import time
from tqdm import tqdm
import pickle
import constants
from config import Config
import argparse
from copy import deepcopy

from tensorboard_logger import configure, log_value

def build_demonstrations(agent):
	"""supervised learning with expert moves"""
	constants_hyperparam = constants.constants
	config = Config.parse("../BlockWorldSimulator/Assets/config.txt")
	assert config.data_mode == Config.TRAIN
	train_size = constants_hyperparam["train_size"]	

	for sample_id in tqdm(range(train_size)):
		img_state = collections.deque([], 5)
		init_imgs = agent.model.image_encoder.build_init_images()
		for img in init_imgs:
			img_state.append(img)
		_, _, img, instruction, trajectory = agent.receive_instruction_image()
		img = np.transpose(img, (2,0,1))
		img_state.append(img)
		previous_action = agent.null_previous_action
		instruction_ids = agent.model.seq_encoder.instruction2id(instruction)

		memory = []
		traj_index = 0
		while True:
			action_id = trajectory[traj_index]
			action_msg = agent.action2msg(action_id)
			traj_index += 1
			agent.connection.send_message(action_msg)
			(status_code, reward, new_img, is_reset) = agent.receive_response()
			memory.append(((deepcopy(img_state), instruction_ids, previous_action), action_id, 1.0))
			new_img = np.transpose(new_img, (2,0,1))
			img_state.append(new_img)
			previous_action = agent.decode_action(action_id)

			if agent.message_protocol_kit.is_reset_message(is_reset):
				agent.connection.send_message("Ok-Reset")
				break

		with open('demonstrations.pkl', 'ab') as output:
			pickle.dump(memory, output, pickle.HIGHEST_PROTOCOL)
	print 'Demonstration Saved'


def learning_from_demonstrations(agent):
	parser = argparse.ArgumentParser(description='Supervised Training hyperparameters')
	parser.add_argument('-batch_size', type=int, default=32, help='batch size for demonstrations')
	parser.add_argument('-max_epochs', type=int, default=1, help='training epochs')
	parser.add_argument('-lr', type=float, default=0.001, help='learning rate')
	parser.add_argument('-entropy_weight', type=float, default=0.1, help='weight for entropy loss')
	parser.add_argument('-replay_memory_size', type=int, default=3200, help='random shuffle')
	args = parser.parse_args()
	batch_size = args.batch_size
	max_epochs = args.max_epochs
	lr = args.lr

	parameters = agent.policy_model.parameters()
	optimizer = torch.optim.Adam(parameters, lr=lr)

	# configure("../runs/" + 'sl_batch_' +str(batch_size) + 'epochs_' + str(max_epochs) + 'lr_' + str(lr) + '_entropy_' + str(args.entropy_weight), flush_secs=2)

	num_experiences = 184131

	step = 0
	for epoch in range(max_epochs):
		f = open('../demonstrations.pkl', 'rb')
		replay_memory = []
		replay_memory_size = args.replay_memory_size
		print 'Leaning from demonstrations Epoch %d' % epoch
		exp_used = 0 # record how many experience used
		memory = []
		# refill the replay memory
		while exp_used < num_experiences:
			if num_experiences - exp_used < replay_memory_size:
				replay_memory_size = num_experiences - exp_used
			while len(replay_memory) < replay_memory_size:
				# print 'refill the replay memory'
				if len(memory) == 0:
					memory = pickle.load(f)
				exp = memory.pop(0)
				replay_memory.append(exp)
			num_batches = (len(replay_memory) - 1) / args.batch_size + 1
			for batch_index in range(num_batches):

				if batch_index == num_batches - 1:
					end = len(replay_memory)
				else:
					end = batch_size * (batch_index + 1)

				imgs, instructions, lens, previous, blocks, directions = agent.exps_to_batchs(replay_memory[(batch_size*batch_index):end])

				# log_value('avg_batch_loss', batch_loss.data.cpu().numpy() / batch_size, step)
				batch_loss = agent.policy_model.sl_loss((imgs, instructions, lens, previous, blocks, directions), args.entropy_weight)
				optimizer.zero_grad()
				batch_loss.backward()
				optimizer.step()
				step += 1
					# log_value('avg_batch_loss', batch_loss.data.cpu().numpy(), step)

			replay_memory = [] # reset the replay memory after use
			exp_used += replay_memory_size


	# save the model
	savepath = '../models/batch_' +str(batch_size) + 'epochs_' + str(max_epochs) + 'lr_' + str(lr) + 'entropy_' + str(args.entropy_weight) + '.pth'
	torch.save(agent.policy_model.state_dict(), savepath)
	print 'Model saved'


if __name__ == '__main__':
	agent = Inverse_agent()
	agent.policy_model.cuda()
	# build_demonstrations(agent)
	learning_from_demonstrations(agent)
	# advesarial_imitation(agent)