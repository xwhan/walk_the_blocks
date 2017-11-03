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

def advesarial_imitation(agent):
	parser = argparse.ArgumentParser(description='Inverse Training hyperparameters')
	parser.add_argument('-max_epochs', type=int, default=2, help='training epochs')
	parser.add_argument('-lr_critic', type=float, default=0.001, help='learning rate')
	parser.add_argument('-lr_agent', type=float, default=0.001)
	parser.add_argument('-entropy_coef', type=float, default=0.1, help='weight for entropy loss')
	parser.add_argument('-clip_epsilon', type=float, default=0.2)
	parser.add_argument('-ppo_epoch', type=int, default=4)
	args = parser.parse_args()

	opti_critic = torch.optim.Adam(agent.critic_model.parameters(), lr=args.lr_critic)
	opti_agent = torch.optim.Adam(agent.policy_model.parameters(), lr=args.lr_agent)

	# configure("runs/" + 'gail_epochs_' + str(args.max_epochs) + '_lr1_' + str(args.lr_agent) + '_lr2_' + str(args.lr_critic) + '_entropyCoef_' + str(args.entropy_coef) + '_clipEpsilon_' + str(args.clip_epsilon) + '_ppoEpoch_' + str(args.ppo_epoch), flush_secs=2)

	constants_hyperparam = constants.constants
	config = Config.parse("../../simulator2/Assets/config.txt")
	assert config.data_mode == Config.TRAIN
	dataset_size = constants_hyperparam["train_size"]

	for epoch in range(args.max_epochs):
		print 'Training Epoch: %d' % epoch
		print 'file holder for demonstrations'

		for sample_id in tqdm(range(dataset_size)):
			img_state = collections.deque([],5)
			init_imgs = agent.policy_model.image_encoder.build_init_images()
			for img in init_imgs:
				img_state.append(img)

			(status_code, bisk_metric, img, instruction, trajectory) = agent.receive_instruction_image()
			img = np.transpose(img, (2,0,1))
			img_state.append(img)
			previous_direction = agent.null_previous_direction
			instruction_ids = agent.policy_model.seq_encoder.instruction2id(instruction)
			state = (img_state, instruction_ids, previous_direction)
			inputs = agent.build_batch_inputs([(state, 0, 0)])

			# direction_prob, block_prob, _ = agent.policy_model(inputs)
			gold_block_id = trajectory[0] / 4
			# block_loss = -torch.log(block_prob.squeeze()[gold_block_id] + 1e-6)
			# opti_agent.zero_grad()
			# block_loss.backward()
			# opti_agent.step()
			
			# sample a trajectory from policy network, collect loss
			sampled_path = []
			while True:
				direction_prob, _, _ = agent.policy_model(inputs)
				direction_id = agent.sample_policy(direction_prob)
				sampled_path.append((deepcopy(state), gold_block_id, direction_id))
				action_msg = agent.action2msg(gold_block_id, direction_id)
				agent.connection.send_message(action_msg)
				(_, reward, new_img, is_reset) = agent.receive_response()
				new_img = np.transpose(new_img, (2,0,1))
				img_state.append(new_img)

				previous_direction = direction_id
				state = (img_state, instruction_ids, previous_direction)
				inputs = agent.build_batch_inputs([(state, 0, 0)])
				if agent.message_protocol_kit.is_reset_message(is_reset):
					agent.connection.send_message('Ok-Reset')
					break
			# log_value('path length', len(sampled_path), sample_id)

			# sample expert demonstrations
			expert_path = []
			expert_memory = pickle.load(f)
			for exp in expert_memory:
				state = exp[0]
				last_direction = state[2][0]
				state_ = (state[0], state[1], last_direction)
				action_id = exp[1]
				block_id = action_id / 4
				if action_id == 80:
					direction_id = 4
					block_id = exp[0][2][1]
				else:
					direction_id = action_id % 4
				expert_path.append((deepcopy(state_), block_id, direction_id))

			# build policy and expert batch
			expert_batch = agent.build_batch_inputs(expert_path)
			policy_batch = agent.build_batch_inputs(sampled_path)

			# discriminator update
			policy_critic_values = agent.critic_model([policy_batch[0], policy_batch[1], policy_batch[2], policy_batch[3]])
			batch_size = policy_critic_values.size()[0]
			gather_indices = torch.arange(0, batch_size).long().cuda() * 5 + policy_batch[5]
			values_selected = policy_critic_values.view(-1)[gather_indices]
			loss_policy = - torch.log(1 - values_selected + 1e-6).mean()

			expert_critic_values = agent.critic_model((expert_batch[0], expert_batch[1], expert_batch[2], expert_batch[3]))
			batch_size = expert_critic_values.size()[0]
			gather_indices = torch.arange(0, batch_size).long().cuda() * 5 + expert_batch[5]
			values_selected = expert_critic_values.view(-1)[gather_indices]
			loss_expert = - torch.log(values_selected + 1e-6).mean() # values is equal to rewards
			d_loss = loss_expert + loss_policy

			opti_critic.zero_grad()
			d_loss.backward()
			# nn.utils.clip_grad_norm(agent.critic_model.parameters(), 5.0)
			opti_critic.step()

			# PPO / TRPO update
			old_model = deepcopy(agent.policy_model)
			old_model.load_state_dict(agent.policy_model.state_dict())
			for _ in range(args.ppo_epoch):
				action_log_probs, dist_entropy = agent.policy_model.evaluate_action((policy_batch[0], policy_batch[1], policy_batch[2], policy_batch[3]), policy_batch[5])
				action_log_probs_old, _ = old_model.evaluate_action((policy_batch[0], policy_batch[1], policy_batch[2], policy_batch[3]), policy_batch[5])
				ratio = torch.exp(action_log_probs - Variable(action_log_probs_old.data))
				values = agent.critic_model((policy_batch[0], policy_batch[1], policy_batch[2], policy_batch[3]))
				batch_size = values.size()[0]
				gather_indices = torch.arange(0, batch_size).long().cuda() * 5 + policy_batch[5]
				values = torch.log(values.view(-1)[gather_indices] + 1e-6) # cost
				target_values = Variable(values.data - 0.5)
				surr1 = ratio * target_values
				surr2 = torch.clamp(ratio, 1.0 - args.clip_epsilon, 1.0 + args.clip_epsilon) * target_values

				# imitation loss
				sl_loss = agent.policy_model.sl_loss(expert_batch, args.entropy_coef)

				# calculate block loss
				block_loss = agent.policy_model.block_loss(policy_batch)

				final_loss = (- torch.min(surr1, surr2).mean() - args.entropy_coef * dist_entropy)*0.1 + block_loss + sl_loss
				opti_agent.zero_grad()
				final_loss.backward()
				# torch.nn.utils.clip_grad_norm(agent.policy_model.parameters(), 5.0)
				opti_agent.step()
			# log_value('policy_entropy', dist_entropy.data.cpu().numpy(), sample_id)

	savepath_1 = '../models/gail_epochs_' + str(args.max_epochs) + '_lr1_' + str(args.lr_agent) + '_lr2_' + str(args.lr_critic) + '_entropyCoef_' + str(args.entropy_coef) + '_clipEpsilon_' + str(args.clip_epsilon) + '_ppoEpoch_' + str(args.ppo_epoch) + '_block_loss_agent.pth'
	savepath_2 = '../models/gail_epochs_' + str(args.max_epochs) + '_lr1_' + str(args.lr_agent) + '_lr2_' + str(args.lr_critic) + '_entropyCoef_' + str(args.entropy_coef) + '_clipEpsilon_' + str(args.clip_epsilon) + '_ppoEpoch_' + str(args.ppo_epoch) + '_block_loss_critic.pth'
	torch.save(agent.policy_model.state_dict(), savepath_1)
	torch.save(agent.critic_model.state_dict(), savepath_2)

if __name__ == '__main__':
	agent = Inverse_agent()
	agent.policy_model.cuda()
	agent.critic_model.cuda()
	# agent.policy_model.load_state_dict(torch.load('../models/best.pth'))
	# agent.critic_model.load_state_dict(torch.load('../models/gail_epochs_2_lr1_0.001_lr2_0.001_entropyCoef_0.1_clipEpsilon_0.2_ppoEpoch_2_block_loss_critic.pth'))
	advesarial_imitation(agent)




