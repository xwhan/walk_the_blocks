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

from tensorboard_logger import configure, log_value

# Define all the training 
def sl_train(agent, max_epochs, train_size, lr):
	"""supervised learning with expert moves"""
	parameters = agent.model.parameters()
	optimizer = torch.optim.Adam(parameters, lr=lr)

	all_demonstrations = []
	for epoch in range(max_epochs):
		print 'Epoch %d' % epoch
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
			inputs = agent.build_inputs(img_state, instruction_ids, previous_action)

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
				# inputs = agent.build_inputs(img_state, instruction_ids, previous_action)

				if agent.message_protocol_kit.is_reset_message(is_reset):
					agent.connection.send_message("Ok-Reset")
					break

			all_demonstrations.append(memory)
			# loss = 0.0
			# np.random.shuffle(memory)
			# episode_len = len(memory)
			# for exp in memory:
			# 	action_prob = agent.model(exp[0]).squeeze()
			# 	loss += - action_prob[exp[1]]
			# loss = loss / episode_len
			# agent.model.zero_grad()
			# loss.backward()
			# torch.nn.utils.clip_grad_norm(parameters, 5.0)
			# optimizer.step()

	# save all the demonstrations
	with open('demonstrations.pkl', 'wb') as output:
		pickle.dump(all_demonstrations, output, pickle.HIGHEST_PROTOCOL)
	print 'Demonstration Saved'
	# save the model
	# torch.save(agent.model.state_dict(), 'models/sl_agent.pth')
	# print 'Model saved'


if __name__ == '__main__':
	constants_hyperparam = constants.constants
	config = Config.parse("../BlockWorldSimulator/Assets/config.txt")
	assert config.data_mode == Config.TRAIN
	dataset_size = constants_hyperparam["train_size"]

	agent = Inverse_agent()
	agent.model.cuda()
	sl_train(agent, 1, dataset_size, 0.001)





