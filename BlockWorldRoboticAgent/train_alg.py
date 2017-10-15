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

from hyperparas import entropy_loss_weight

from tensorboard_logger import configure, log_value

# Define all the training 
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

def cal_entropy(action_prob):
	action_prob = action_prob.data.cpu().numpy()
	entropy = 0
	for prob in action_prob:
		entropy += - prob * np.log(prob + 1e-13)
	return entropy

def learning_from_demonstrations(agent):
	parser = argparse.ArgumentParser(description='Supervised Training hyperparameters')
	parser.add_argument('-batch_size', type=int, default=32, help='batch size for demonstrations')
	parser.add_argument('-max_epochs', type=int, default=1, help='training epochs')
	parser.add_argument('-lr', type=float, default=0.001, help='learning rate')
	parser.add_argument('-entropy_weight', type=float, default=0.1, help='weight for entropy loss')
	parser.add_argument('-replay_memory_size', type=int, default=3200, help='random shuffle')
	parser.add_argument('-clip_value', type=float, default=5.0)

	args = parser.parse_args()
	batch_size = args.batch_size
	max_epochs = args.max_epochs
	lr = args.lr
	entropy_loss_weight = args.entropy_weight

	parameters = agent.policy_model.parameters()
	optimizer = torch.optim.Adam(parameters, lr=lr)

	configure("runs/" + 'sl_batch_' +str(batch_size) + 'epochs_' + str(max_epochs) + 'lr_' + str(lr) + 'entropy_' + str(entropy_loss_weight), flush_secs=2)

	num_experiences = 184131

	step = 0
	for epoch in range(max_epochs):
		f = open('demonstrations.pkl', 'rb')
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

			print 'shuffle all experinces in the replay memory'
			np.random.shuffle(replay_memory)
			num_batches = (len(replay_memory) - 1) / args.batch_size + 1
			for batch_index in tqdm(range(num_batches)):
				loss = 0.0
				entropy_loss = 0.0

				if batch_index == num_batches - 1:
					end = len(replay_memory)
				else:
					end = batch_size * (batch_index + 1)

				for exp in replay_memory[(batch_size*batch_index):end]:
					state = exp[0]
					inputs = agent.build_batch_inputs([(state, 0)])
					action_prob = agent.policy_model(inputs).squeeze()
					prob_entropy_neg = torch.sum(action_prob * torch.log(action_prob + 1e-13)) 
					entropy_loss += prob_entropy_neg
					loss += - torch.log(action_prob[exp[1]] + 1e-13)

				final_loss = (loss + entropy_loss_weight * entropy_loss) / batch_size
				log_value('avg_batch_loss', loss.data.cpu().numpy() / batch_size, step)
				log_value('avg_batch_entropy_loss', entropy_loss.data.cpu().numpy() / batch_size, step)

				optimizer.zero_grad()
				loss.backward()
				torch.nn.utils.clip_grad_norm(parameters, 5.0)
				optimizer.step()
				step += 1

			replay_memory = [] # reset the replay memory after use
			exp_used += replay_memory_size


	# save the model
	savepath = 'models/sl_' +  'batch_' +str(batch_size) + 'epochs_' + str(max_epochs) + 'lr_' + str(lr) + 'entropy_' + str(entropy_loss_weight) + '_clip_' + str(args.clip_value) + '.pth'
	torch.save(agent.policy_model.state_dict(), savepath)
	print 'Model saved'

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

	configure("runs/" + 'epochs_' + str(args.max_epochs) + '_lr1_' + str(args.lr_agent) + '_lr2_' + str(args.lr_critic) + '_entropyCoef_' + str(args.entropy_coef) + '_clipEpsilon_' + str(args.clip_epsilon) + '_ppoEpoch_' + str(args.ppo_epoch), flush_secs=2)

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
			previous_action = agent.null_previous_action
			instruction_ids = agent.policy_model.seq_encoder.instruction2id(instruction)
			state = (img_state, instruction_ids, previous_action)
			inputs = agent.build_batch_inputs([(state, 0)])
			
			# sample a trajectory from policy network, collect loss
			sampled_path = []
			while True:
				action_prob = agent.policy_model(inputs)
				action_prob = action_prob.squeeze()
				action_id = agent.sample_policy(action_prob)
				sampled_path.append((deepcopy(state), action_id))
				action_msg = agent.action2msg(action_id)
				agent.connection.send_message(action_msg)
				(_, reward, new_img, is_reset) = agent.receive_response()
				new_img = np.transpose(new_img, (2,0,1))
				img_state.append(new_img)
				previous_action = agent.decode_action(action_id)
				state = (img_state, instruction_ids, previous_action)
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
				action_id = exp[1]
				expert_path.append((deepcopy(state), action_id))

			# build policy and expert batch
			expert_batch = agent.build_batch_inputs(expert_path)
			policy_batch = agent.build_batch_inputs(sampled_path)

			# discriminator update
			policy_critic_values = agent.critic_model((policy_batch[0], policy_batch[1], policy_batch[2], policy_batch[3]))
			batch_size = policy_critic_values.size()[0]
			gather_indices = torch.arange(0, batch_size).long().cuda() * batch_size + policy_batch[4].data
			values_selected = policy_critic_values.view(-1).index_select(0, Variable(gather_indices))
			loss_policy = torch.log(values_selected + 1e-13).mean()

			expert_critic_values = agent.critic_model((expert_batch[0], expert_batch[1], expert_batch[2], expert_batch[3]))
			batch_size = expert_critic_values.size()[0]
			gather_indices = torch.arange(0, batch_size).long().cuda() * batch_size + expert_batch[4].data
			values_selected = expert_critic_values.view(-1).index_select(0, Variable(gather_indices))
			loss_expert = torch.log(1 - values_selected + 1e-13).mean()
			d_loss =  - (loss_expert + loss_policy)

			opti_critic.zero_grad()
			d_loss.backward()
			# torch.nn.utils.clip_grad_norm(parameters, 5.0)
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
				gather_indices = torch.arange(0, batch_size).long().cuda() * batch_size + policy_batch[4].data
				values = torch.log(values.view(-1).index_select(0, Variable(gather_indices)) + 1e-13) # cost
				q_values = - Variable(values.data)
				surr1 = ratio * q_values
				surr2 = torch.clamp(ratio, 1.0 - args.clip_epsilon, 1.0 + args.clip_epsilon) * q_values
				final_loss = - torch.min(surr1, surr2).mean() - args.entropy_coef * dist_entropy
				opti_agent.zero_grad()
				final_loss.backward()
				opti_agent.step()
			log_value('policy_entropy', dist_entropy.data.cpu().numpy(), sample_id)

	savepath_1 = 'models/gail_epochs_' + str(args.max_epochs) + '_lr1_' + str(args.lr_agent) + '_lr2_' + str(args.lr_critic) + '_entropyCoef_' + str(args.entropy_coef) + '_clipEpsilon_' + str(args.clip_epsilon) + '_ppoEpoch_' + str(args.ppo_epoch) + '_agent.pth'
	savepath_2 = 'models/gail_epochs_' + str(args.max_epochs) + '_lr1_' + str(args.lr_agent) + '_lr2_' + str(args.lr_critic) + '_entropyCoef_' + str(args.entropy_coef) + '_clipEpsilon_' + str(args.clip_epsilon) + '_ppoEpoch_' + str(args.ppo_epoch) + '_critic.pth'
	torch.save(agent.policy_model.state_dict(), savepath_1)
	torch.save(agent.critic_model.state_dict(), savepath_2)

if __name__ == '__main__':
	agent = Inverse_agent()
	agent.policy_model.cuda()
	# agent.critic_model.cuda()
	# build_demonstrations(agent)
	learning_from_demonstrations(agent)
	# advesarial_imitation(agent)




