#!/usr/bin/python 

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torch.nn.init as init
from agent import *
from policy_model import *
from copy import deepcopy
import constants
from config import Config
import collections
import numpy as np

from tensorboard_logger import configure, log_value

def ppo_update(agent, sl_path):
	parser = argparse.ArgumentParser(description='PPO update')
	parser.add_argument('-max_epochs', type=int, default=1, help='training epochs')
	parser.add_argument('-lr', type=float, default=0.001, help='learning rate')
	parser.add_argument('-ppo_epoch', type=int, default=4)
	parser.add_argument('-clip_epsilon', type=float, default=0.2)
	parser.add_argument('-entropy_coef', type=float, default=0.1, help='weight for entropy loss')
	args = parser.parse_args()

	opti = torch.optim.Adam(agent.policy_model.parameters(), lr=args.lr)

	# load from best sl model
	agent.policy_model.cuda()
	agent.policy_model.load_state_dict(torch.load(sl_path))

	# reinitialize direction layer
	nn.init.xavier_uniform(agent.policy_model.direction_layer, gain=nn.init.calculate_gain('relu'))

	constants_hyperparam = constants.constants
	config = Config.parse("../BlockWorldSimulator/Assets/config.txt")
	assert config.data_mode == Config.TRAIN
	dataset_size = constants_hyperparam["train_size"]

	for epoch in range(args.max_epochs):

		for sample_id in tqdm(range(dataset_size)):
			img_state = collections.deque([], 5)
			init_imgs = agent.policy_model.image_encoder.build_init_images()
			for img in init_imgs:
				img_state.append(img)

			(_, _, img, instruction, traj) = agent.receive_instruction_image()
			img = np.transpose(img, (2,0,1))
			img_state.append(img)
			previous_direction = agent.null_previous_direction
			instruction_ids = agent.policy_model.seq_encoder.instruction2id(instruction)
			state = (img_state, instruction_ids, previous_direction)
			inputs = agent.build_batch_inputs([(state)])

			replay_memory = []
			steps = 0
			rewards = []
			baselines = []

			# roll out
			while True:
				d_probs, b_probs, _ = agent.policy_model(inputs)
				d_id = agent.sample_policy(d_probs, method='random')
				b_id = agent.sample_policy(b_probs, method='greedy')
				action_msg = agent.action2msg(b_id, d_id)
				agent.connection.send_message(action_msg)

				(_, reward, new_img, is_reset) = agent.receive_response_and_image()
				rewards.append(reward)
				replay_memory_item = (deepcopy(state), reward, b_id, d_id)
				replay_memory.append(replay_memory_item)

				img_state.append(new_img)
				previous_direction = 



