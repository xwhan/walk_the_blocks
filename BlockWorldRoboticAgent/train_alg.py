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

from hyperparas import entropy_loss_weight

from tensorboard_logger import configure, log_value

# Define all the training 
def build_demonstrations(agent, max_epochs, train_size, lr):
	"""supervised learning with expert moves"""

	all_demonstrations = []

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
			memory.append(((img_state, instruction_ids, previous_action), action_id, 1.0))
			new_img = np.transpose(new_img, (2,0,1))
			img_state.append(new_img)
			previous_action = agent.decode_action(action_id)

			if agent.message_protocol_kit.is_reset_message(is_reset):
				agent.connection.send_message("Ok-Reset")
				break

		all_demonstrations.append(memory)

	with open('demonstrations.pkl', 'wb') as output:
		pickle.dump(all_demonstrations, output, pickle.HIGHEST_PROTOCOL)
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

	print 'loading demonstrations'
	all_demonstrations = pickle.load(open('demonstrations.pkl', 'rb'))
	all_experince = []

	for memory in all_demonstrations:
		for exp in memory:
			all_experince.append(exp)
	num_experiences = len(all_experince)
	num_batches = (num_experiences - 1) / batch_size + 1

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
	parser = argparse.ArgumentParser(description='Supervised Training hyperparameters')
	parser.add_argument('-max_epochs', type=int, default=1, help='training epochs')
	parser.add_argument('-lr', type=float, default=0.001, help='learning rate')
	parser.add_argument('-entropy_weight', type=float, default=0.1, help='weight for entropy loss')
	args = parser.parse_args()

	constants_hyperparam = constants.constants
	config = Config.parse("../BlockWorldSimulator/Assets/config.txt")
	assert config.data_mode == Config.TRAIN
	dataset_size = constants_hyperparam["train_size"]	

	for sample_id in tqdm(range(dataset_size)):
		img_state = collections.deque([],5)
		init_imgs = agent.model.image_encoder.build_init_images()
		for img in init_imgs:
			img_state.append(img)
		(status_code, bisk_metric, img, instruction, trajectory) = agent.receive_instruction_image()
		img = np.transpose(img, (2,0,1))
		img_state.append(img)
		previous_action = agent.null_previous_action
		instruction_ids = agent.model.seq_encoder.instruction2id(instruction)
		inputs = agent.build_inputs(img_state, instruction_ids, previous_action)

		


if __name__ == '__main__':
	


	agent = Inverse_agent()
	agent.model.cuda()
	learning_from_demonstrations(agent)





