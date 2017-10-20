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

def advesarial_imitation(agent):
	parser = argparse.ArgumentParser(description='Inverse Training hyperparameters')
	parser.add_argument('-max_epochs', type=int, default=1, help='training epochs')
	parser.add_argument('-lr_critic', type=float, default=0.001, help='learning rate')
	parser.add_argument('-lr_agent', type=float, default=0.001)
	parser.add_argument('-entropy_coef', type=float, default=0.1, help='weight for entropy loss')
	parser.add_argument('-clip_epsilon', type=float, default=0.2)
	parser.add_argument('-ppo_epoch', type=int, default=4)
	args = parser.parse_args()

	opti_critic = torch.optim.Adam(agent.critic_model.parameters(), lr=args.lr_critic)
	opti_agent = torch.optim.Adam(agent.policy_model.parameters(), lr=args.lr_agent)

	configure("runs/" + 'broken_epochs_' + str(args.max_epochs) + '_lr1_' + str(args.lr_agent) + '_lr2_' + str(args.lr_critic) + '_entropyCoef_' + str(args.entropy_coef) + '_clipEpsilon_' + str(args.clip_epsilon) + '_ppoEpoch_' + str(args.ppo_epoch), flush_secs=2)

	constants_hyperparam = constants.constants
	config = Config.parse("../BlockWorldSimulator/Assets/config.txt")
	assert config.data_mode == Config.TRAIN
	dataset_size = constants_hyperparam["train_size"]

	for epoch in range(args.max_epochs):
		print 'Training Epoch: %d' % epoch
		print 'file holder for demonstrations'
		f = open('demonstrations.pkl', 'rb')

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
			real_img_state = []
			for _ in range(len(img_state) - 1):
				real_img_state.append(img_state[-1] - img_state[_])
			real_img_state.append(img_state[-1])
			state = (real_img_state, instruction_ids, previous_direction)
			inputs = agent.build_batch_inputs([(state, 0)])

			_, block_prob = agent.policy_model(inputs)
			gold_block_id = trajectory[0] / 4
			block_loss = -torch.log(block_prob.squeeze()[gold_block_id] + 1e-8)
			opti_agent.zero_grad()
			block_loss.backward()
			opti_agent.step()
			
			# sample a trajectory from policy network, collect loss
			sampled_path = []
			while True:
				direction_prob, _ = agent.policy_model(inputs)
				direction_id = agent.sample_policy(direction_prob)
				sampled_path.append((deepcopy(state), direction_id))
				if direction_id == 4:
					action_id = 80
				else:
					action_id = gold_block_id * agent.num_direction + direction_id
				action_msg = agent.action2msg(action_id)
				agent.connection.send_message(action_msg)
				(_, reward, new_img, is_reset) = agent.receive_response()
				new_img = np.transpose(new_img, (2,0,1))
				img_state.append(new_img)
				real_img_state = []
				for _ in range(len(img_state) - 1):
					real_img_state.append(img_state[-1] - img_state[_])
				real_img_state.append(img_state[-1])
				state = (real_img_state, instruction_ids, previous_direction)
				inputs = agent.build_batch_inputs([(state, 0)])
				if agent.message_protocol_kit.is_reset_message(is_reset):
					agent.connection.send_message('Ok-Reset')
					break
			log_value('path length', len(sampled_path), sample_id)

			# sample expert demonstrations
			expert_path = []
			expert_memory = pickle.load(f)
			for exp in expert_memory:
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

			# build policy and expert batch
			expert_batch = agent.build_batch_inputs(expert_path)
			policy_batch = agent.build_batch_inputs(sampled_path)

			# discriminator update
			policy_critic_values = agent.critic_model([policy_batch[0], policy_batch[1], policy_batch[2], policy_batch[3]])
			batch_size = policy_critic_values.size()[0]
			gather_indices = (torch.arange(0, batch_size).long() * (agent.num_direction+1)).cuda() + policy_batch[4].data
			values_selected = policy_critic_values.view(-1)[gather_indices]
			loss_policy = torch.log(values_selected + 1e-8).mean()

			expert_critic_values = agent.critic_model((expert_batch[0], expert_batch[1], expert_batch[2], expert_batch[3]))
			batch_size = expert_critic_values.size()[0]
			gather_indices = (torch.arange(0, batch_size).long() * (agent.num_direction+1)).cuda() + expert_batch[4].data
			values_selected = expert_critic_values.view(-1)[gather_indices]
			loss_expert = torch.log(1 - values_selected + 1e-8).mean()
			d_loss =  - (loss_expert + loss_policy)

			opti_critic.zero_grad()
			d_loss.backward()
			opti_critic.step()

			# PPO / TRPO update
			old_model = deepcopy(agent.policy_model)
			old_model.load_state_dict(agent.policy_model.state_dict())
			for _ in range(args.ppo_epoch):
				action_log_probs, dist_entropy = agent.policy_model.evaluate_action((policy_batch[0], policy_batch[1], policy_batch[2], policy_batch[3]), policy_batch[4])
				action_log_probs_old, _ = old_model.evaluate_action((policy_batch[0], policy_batch[1], policy_batch[2], policy_batch[3]), policy_batch[4])
				ratio = torch.exp(action_log_probs - Variable(action_log_probs_old.data))
				values = agent.critic_model((policy_batch[0], policy_batch[1], policy_batch[2], policy_batch[3]))
				batch_size = values.size()[0]
				gather_indices = torch.arange(0, batch_size).long().cuda() * (agent.num_direction + 1) + policy_batch[4].data
				values = torch.log(values.view(-1)[gather_indices] + 1e-8) # cost
				q_values = - Variable(values.data)
				surr1 = ratio * q_values
				surr2 = torch.clamp(ratio, 1.0 - args.clip_epsilon, 1.0 + args.clip_epsilon) * q_values
				final_loss = - torch.min(surr1, surr2).mean() - args.entropy_coef * dist_entropy
				opti_agent.zero_grad()
				final_loss.backward()
				# torch.nn.utils.clip_grad_norm(agent.policy_model.parameters(), 5.0)
				opti_agent.step()
			log_value('policy_entropy', dist_entropy.data.cpu().numpy(), sample_id)

	savepath_1 = 'models/broken_gail_epochs_' + str(args.max_epochs) + '_lr1_' + str(args.lr_agent) + '_lr2_' + str(args.lr_critic) + '_entropyCoef_' + str(args.entropy_coef) + '_clipEpsilon_' + str(args.clip_epsilon) + '_ppoEpoch_' + str(args.ppo_epoch) + '_agent.pth'
	savepath_2 = 'models/broken_gail_epochs_' + str(args.max_epochs) + '_lr1_' + str(args.lr_agent) + '_lr2_' + str(args.lr_critic) + '_entropyCoef_' + str(args.entropy_coef) + '_clipEpsilon_' + str(args.clip_epsilon) + '_ppoEpoch_' + str(args.ppo_epoch) + '_critic.pth'
	torch.save(agent.policy_model.state_dict(), savepath_1)
	torch.save(agent.critic_model.state_dict(), savepath_2)

if __name__ == '__main__':
	agent = Inverse_agent()
	agent.policy_model.cuda()
	agent.critic_model.cuda()
	advesarial_imitation(agent)




