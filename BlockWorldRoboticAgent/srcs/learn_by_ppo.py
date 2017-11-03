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
	parser.add_argument('-lr', type=float, default=0.0001, help='learning rate')
	parser.add_argument('-ppo_epoch', type=int, default=4)
	parser.add_argument('-clip_epsilon', type=float, default=0.05)
	parser.add_argument('-entropy_coef', type=float, default=0.1, help='weight for entropy loss')
	args = parser.parse_args()

	# configure("runs/" + 'ppo_critic_epochs_' + str(args.max_epochs) + '_lr_' + str(args.lr) + '_entropyCoef_' + str(args.entropy_coef) + '_clipEpsilon_' + str(args.clip_epsilon) + '_ppoEpoch_' + str(args.ppo_epoch), flush_secs=2)

	# configure("runs/" + 'from_scratch_ppo_critic_epochs_' + str(args.max_epochs) + '_lr_' + str(args.lr) + '_entropyCoef_' + str(args.entropy_coef) + '_clipEpsilon_' + str(args.clip_epsilon) + '_ppoEpoch_' + str(args.ppo_epoch), flush_secs=2)

	opti = torch.optim.Adam(agent.policy_model.parameters(), lr=args.lr)

	# load from best sl model
	# agent.policy_model.load_state_dict(torch.load(sl_path))

	# reinitialize direction layer
	# nn.init.xavier_uniform(agent.policy_model.direction_layer.weight, gain=nn.init.calculate_gain('relu'))

	constants_hyperparam = constants.constants
	config = Config.parse("../../BlockWorldSimulator/Assets/config.txt")
	assert config.data_mode == Config.TRAIN
	dataset_size = constants_hyperparam["train_size"]

	step = 0

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
			replay_memory = []
			steps = 0
			rewards = []
			baselines = []

			# roll out
			while True:
				d_probs, b_probs, baseline = agent.policy_model(inputs)
				d_id = agent.sample_policy(d_probs, method='random')
				# b_id = agent.sample_policy(b_probs, method='greedy')
				baseline = baseline.squeeze()
				baselines.append(baseline.data.cpu().numpy()[0])
				b_id = gold_block_id
				action_msg = agent.action2msg(b_id, d_id)
				agent.connection.send_message(action_msg)

				(_, reward, new_img, is_reset) = agent.receive_response()
				new_img = np.transpose(new_img, (2,0,1))
				rewards.append(reward)
				replay_memory_item = (deepcopy(state), b_id, d_id)
				replay_memory.append(replay_memory_item)

				if agent.message_protocol_kit.is_reset_message(is_reset):
					agent.connection.send_message('Ok-Reset')
					break	

				img_state.append(new_img)
				previous_direction = d_id
				state = (img_state, instruction_ids, previous_direction)
				inputs = agent.build_batch_inputs([(state, 0, 0)])

			# rewards_final = [0] * len(rewards)
			# for _ in range(len(rewards)):
			# 	rewards_final[_] = sum(rewards[_:])
			batch = agent.build_batch_inputs(replay_memory)

			# sample expert demonstration
			expert_path = []
			expert_memory = pickel.load(f)
			for exp in expert_memory
				state = exp[0]
				last_direction = state[2][0]
				real_image = []
				for _ in range(len(state[0]) - 1):
					real_image.append(state[0][-1] - state[0][_])
				real_image.append(state[0][-1])
				state_ = (real_image, state[1], last_direction)
				action_id = exp[1]
				if action_id == 80:
					direction_id = 4
				else:
					direction_id = action_id % 4
				expert_path.append((deepcopy(state_), direction_id))
			expert_batch = agent.build_batch_inputs(expert_path)

			old_model = deepcopy(agent.policy_model)
			old_model.load_state_dict(agent.policy_model.state_dict())
			for _ in range(args.ppo_epoch):
				ppo_loss = agent.policy_model.ppo_loss(batch, old_model, rewards, baselines, args)
				sl_loss = 
				opti.zero_grad()
				final_loss.backward()
				# nn.utils.clip_grad_norm(agent.policy_model.parameters(), 5.0)
				opti.step()
				step += 1
				# log_value('ppo loss', final_loss.data.cpu().numpy(), step)

			# log_value('path length', len(rewards), sample_id)

	save_path = '../models/from_scratch_ppo_critic_epochs' + str(args.max_epochs) + '_lr_' + str(args.lr) + '_clip_' + str(args.clip_epsilon) + '.pth'
	torch.save(agent.policy_model.state_dict(), save_path)

if __name__ == '__main__':
	agent = Inverse_agent()
	agent.policy_model.cuda()
	ppo_update(agent, '../models/sl_best.pth')
