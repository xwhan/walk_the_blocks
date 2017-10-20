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


def reward_learner(agent):
	"""
	learning to predict the reward signals
	off-policy learning
	maybe use the learned reward for exploration
	"""
	parser = argparse.ArgumentParser(description='model the reward')
	parser.add_argument('-max_epochs', type=int, default=1, help='training epochs')
	parser.add_argument('-lr', type=float, default=0.001, help='learning rate')
	args = parser.parse_args()

	opti_value = torch.optim.Adam(agent.value_model.parameters(), lr=args.lr)

	configure("runs/" + 'value_model_epochs_' + str(args.max_epochs) + '_lr_' + str(args.lr), flush_secs=1)

	constants_hyperparam = constants.constants
	config = Config.parse("../BlockWorldSimulator/Assets/config.txt")
	assert config.data_mode == Config.TRAIN
	dataset_size = constants_hyperparam["train_size"]

	for epoch in range(args.max_epochs):
		print 'Training Epoch: %d' % epoch

		for sample_id in tqdm(range(dataset_size)):
			img_state = collections.deque([],5)
			init_imgs = agent.value_model.image_encoder.build_init_images()
			for img in init_imgs:
				img_state.append(img)
			(status_code, bisk_metric, img, instruction, trajectory) = agent.receive_instruction_image()
			img = np.transpose(img, (2,0,1))
			img_state.append(img)
			previous_direction = agent.null_previous_direction
			instruction_ids = agent.value_model.seq_encoder.instruction2id(instruction)
			state = (img_state, instruction_ids, previous_direction)
			inputs = agent.build_batch_inputs([(state, 0)])

			gold_block_id = trajectory[0] / 4

			loss = 0.0
			steps = 0
			while True:
				values = agent.value_model(inputs)
				values = values.squeeze()
				direction_id = agent.explore(values, 1.0, 0.2)
				if direction_id == 4:
					action_id = 80
				else:
					action_id = gold_block_id * agent.num_direction + direction_id
				action_msg = agent.action2msg(action_id)
				agent.connection.send_message(action_msg)
				(_, reward, new_img, is_reset) = agent.receive_response()
				loss += (values[direction_id] - reward) ** 2
				steps += 1

				new_img = np.transpose(new_img, (2,0,1))
				img_state.append(new_img)
				state = (img_state, instruction_ids, previous_direction)
				inputs = agent.build_batch_inputs([(state, 0)])
				if agent.message_protocol_kit.is_reset_message(is_reset):
					agent.connection.send_message('Ok-Reset')
					break

			final_loss = loss / steps
			log_value('value error', final_loss.data.cpu().numpy(), sample_id)
			opti_value.zero_grad()
			final_loss.backward()
			opti_value.step()

	savepath_1 = 'models/reward_model_epochs_' + str(args.max_epochs) + '_lr_' + str(args.lr) + '_agent.pth'
	torch.save(agent.value_model.state_dict(), savepath_1)

if __name__ == '__main__':
	agent = Inverse_agent()
	agent.value_model.cuda()
	reward_learner(agent)
