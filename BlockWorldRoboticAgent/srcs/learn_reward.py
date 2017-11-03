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
import collections

from tensorboard_logger import configure, log_value

def critic_training(agent, sl_path):
	parser = argparse.ArgumentParser(description='critic training parameters')
	parser.add_argument('-max_epochs', type=int, default=2)
	parser.add_argument('-lr', type=float, default=0.0005)
	parser.add_argument('-replay_memory_size', type=int, default=6400)
	args = parser.parse_args()

	opti_critic = torch.optim.Adam(agent.critic_model.parameters(), lr=args.lr)
	agent.policy_model.load_state_dict(torch.load(sl_path))

	constants_hyperparam = constants.constants
	config = Config.parse("../../BlockWorldSimulator/Assets/config.txt")
	assert config.data_mode == Config.TRAIN
	dataset_size = constants_hyperparam["train_size"]

	for epoch in range(args.max_epochs):
		f = open('../demonstrations.pkl', 'rb')
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
			inputs = agent.build_batch_inputs([(state, 0, 0)])

			gold_block_id = traj[0] / 4
			memory = []
			# sample a path using current policy
			while True:
				d_probs, _, _ = agent.policy_model(inputs)
				d_id = agent.sample_policy(d_probs, method='random')
				action_msg = agent.action2msg(gold_block_id, d_id)
				agent.connection.send_message(action_msg)
				(_, reward, new_img, is_reset) = agent.receive_response()
				new_img = np.transpose(new_img, (2,0,1))
				memory_item = (deepcopy(state), d_id)
				memory.append(memory_item)

				if agent.message_protocol_kit.is_reset_message(is_reset):
					agent.connection.send_message('Ok-Reset')
					break

				img_state.append(new_img)
				previous_direction = d_id
				state = (img_state, instruction_ids, previous_direction)
				inputs = agent.build_batch_inputs([(state, 0, 0)])

			# expert path
			expert_path = pickle.load(f)
			expert_memory = []
			for exp in expert_path:
				if exp[1] == 80:
					gold_direction = 4
				else:
					gold_direction = exp[1] % 4
				state = exp[0]
				imgs = state[0]
				instruciton_ids = state[1]
				previous_direction = state[2][0]
				state = (imgs, instruction_ids, previous_direction)
				memory_item = (deepcopy(state), gold_direction)
				expert_memory.append(memory_item)

			final_loss = agent.critic_model.inverse_loss(memory, expert_memory)
			opti_critic.zero_grad()
			final_loss.backward()
			nn.utils.clip_grad_norm(agent.policy_model.parameters(), 5.0)
			opti_critic.step()

			# update_policy_with_critic(agent)

	save_path = '../models/inverse_critic_with_sl_policy.pth'
	torch.save(agent.critic_model.state_dict(), save_path)

if __name__ == '__main__':
	agent = Inverse_agent()
	agent.policy_model.cuda()
	agent.critic_model.cuda()
	critic_training(agent, '../models/sl_best.pth')


