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

	args = parser.parse_args()
	batch_size = args.batch_size
	max_epochs = args.max_epochs
	lr = args.lr
	entropy_loss_weight = args.entropy_weight

	parameters = agent.model.parameters()
	optimizer = torch.optim.Adam(parameters, lr=lr)

	configure("runs/" + 'batch_' +str(batch_size) + 'epochs_' + str(max_epochs) + 'lr_' + str(lr) + 'entropy_' + str(entropy_loss_weight), flush_secs=2)

	f = open('demonstrations.pkl', 'rb')
	num_experiences = 0


	step = 0 # update steps
	for epoch in range(max_epochs):
		print 'Leaning from demonstrations Epoch %d' % epoch
		print 'shuffle all experinces'
		np.random.shuffle(all_experince)

		for batch_index in tqdm(range(num_batches)):
			loss = 0.0
			entropy_loss = 0.0

			if batch_index == num_batches - 1:
				end = num_experiences
			else:
				end = batch_size * (batch_index + 1)

			for exp in all_experince[(batch_size*batch_index):end]:
				state = exp[0]
				inputs = agent.build_inputs(state[0], state[1], state[2])
				action_prob = agent.model(inputs).squeeze()
				prob_entropy_neg = torch.sum(action_prob * torch.log(action_prob + 1e-13)) 
				entropy_loss += prob_entropy_neg
				loss += - torch.log(action_prob[exp[1]] + 1e-13)

			final_loss = loss + entropy_loss_weight * entropy_loss
			log_value('avg_batch_loss', loss.data.cpu().numpy() / batch_size, step)
			log_value('avg_batch_entropy_loss', entropy_loss.data.cpu().numpy() / batch_size, step)
			step += 1

			agent.model.zero_grad()
			loss.backward()
			torch.nn.utils.clip_grad_norm(parameters, 5.0)
			optimizer.step()

	# save the model
	savepath = 'models/sl_' +  'batch_' +str(batch_size) + 'epochs_' + str(max_epochs) + 'lr_' + str(lr) + 'entropy_' + str(entropy_loss_weight) + '.pth'
	torch.save(agent.model.state_dict(), savepath)
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

	constants_hyperparam = constants.constants
	config = Config.parse("../BlockWorldSimulator/Assets/config.txt")
	assert config.data_mode == Config.TRAIN
	dataset_size = constants_hyperparam["train_size"]

	print 'file holder for demonstrations'
	f = open('demonstrations.pkl', 'rb')

	for epoch in range(args.max_epochs):
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
			inputs = agent.build_inputs(img_state, instruction_ids, previous_action)
			state = (img_state, instruction_ids, previous_action)
			
			# sample a trajectory from policy network, collect loss
			sampled_path = []
			policy_log_values = 0.0
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
				inputs = agent.build_inputs(img_state, instruction_ids, previous_action)
				state = (img_state, instruction_ids, previous_action)
				if agent.message_protocol_kit.is_reset_message(is_reset):
					agent.connection.send_message('Ok-Reset')
					break

			# sample expert demonstrations
			expert_path = []
			expert_memory = pickle.load(f)
			for exp in expert_memory:
				state = exp[0]
				action_id = exp[1]
				expert_path.append((deepcopy(state), action_id))

			# discriminator update
			loss_policy = 0.0
			for exp in sampled_path:
				critic_values = agent.critic_model(exp[0]).squeeze()
				loss_policy += torch.log(critic_values[exp[1]] + 1e-13)
			loss_expert = 0.0
			for exp in expert_path:
				critic_values = agent.critic_model(exp[0]).squeeze()
				loss_expert += torch.log(1 - critic_values[exp[1]] + 1e-13)
			d_loss = loss_policy / len(sampled_path) + loss_expert / len(expert_path)
			opti_critic.zero_grad()
			d_loss.backward()
			# torch.nn.utils.clip_grad_norm(parameters, 5.0)
			opti_critic.step()

			# PPO / TRPO update
			old_model = deepcopy(agent.policy_model)
			old_model.load_state_dict(agent.policy_model.state_dict())
			for _ in range(args.ppo_epoch):
				final_loss = 0.0
				for exp in sampled_path:
					action_log_prob, dist_entropy = agent.policy_model.evaluate_action(exp[0], exp[1])
					action_log_prob_old, _ = old_model.evaluate_action(exp[0], exp[1])
					ratio = torch.exp(action_log_prob - Variable(action_log_prob_old.data))
					value = agent.critic_model(exp[0]).squeeze()[exp[1]]
					value = torch.log(value + 1e-13)
					q_value = - Variable(value.data)
					surr1 = ratio * q_value
					surr2 = torch.clamp(ratio, 1.0 - args.clip_epsilon, 1.0 + args.clip_epsilon) * q_value
					agent_loss = - torch.min(surr1, surr2)
					final_loss += agent_loss - args.entropy_coef * dist_entropy
				opti_agent.zero_grad()
				opti_agent.step()

	savepath_1 = 'models/gail_agent_v0.0.pth'
	savepath_2 = 'models/gail_agent_v0.1.pth'
	torch.save(agent.policy_model.state_dict(), savepath_1)
	torch.save(agent.critic_model.state_dict(), savepath_2)

if __name__ == '__main__':
	agent = Inverse_agent()
	agent.policy_model.cuda()
	agent.critic_model.cuda()
	# build_demonstrations(agent)
	# learning_from_demonstrations(agent)
	advesarial_imitation(agent)




