#!/usr/bin/python

import message_protocol_util as mpu
import reliable_connect as rc
import sys
import logger
from config import Config
import collections
import numpy as np
import time

from hyperparas import *
from attention import *

class Inverse_agent(object):
	"""inverse rl with GAN"""
	def __init__(self):		
		logger.Log.open('./log.txt')

		# Connect to simulator
		if len(sys.argv) < 2:
			logger.Log.info("IP not given. Using localhost i.e. 0.0.0.0")
			self.unity_ip = "0.0.0.0"
		else:
			self.unity_ip = sys.argv[1]

		if len(sys.argv) < 3:
			logger.Log.info("PORT not given. Using 11000")
			self.PORT = 11000
		else:
			self.PORT = int(sys.argv[2])

		# Size of image
		config = Config.parse("../BlockWorldSimulator/Assets/config.txt")
		self.config = config
		image_dim = self.config.screen_size

		self.connection = rc.ReliableConnect(self.unity_ip, self.PORT, image_dim)
		self.connection.connect()

		# Dataset specific parameters
		self.num_block = 20
		self.num_direction = 4
		use_stop = True
		if use_stop:
			self.num_actions = self.num_block * self.num_direction + 1  # 1 for stopping
		else:
			self.num_actions = self.num_block * self.num_direction

		# Create toolkit of message protocol between simulator and agent
		self.message_protocol_kit = mpu.MessageProtocolUtil(self.num_direction, self.num_actions, use_stop)

		self.null_previous_action = (self.num_direction+1, self.num_block) # (5, 20)

		self.model = Context_attention(image_embed_dim=200, hidden_dim=200, action_dim_1=32, action_dim_2=24, inter_dim=120)

	def receive_instruction_image(self):

		img = self.connection.receive_image()
		response = self.connection.receive_message()
		(status_code, bisk_metric, _, instruction, trajectory) = self.message_protocol_kit.decode_reset_message(response)
		return status_code, bisk_metric, img, instruction, trajectory

	def receive_response(self):
		"""recieve the feedback from the environment after executing one action"""
		img = self.connection.receive_image()
		response = self.connection.receive_message()
		status_code, reward, _, reset_file = self.message_protocol_kit.decode_message(response)
		return status_code, reward, img, reset_file

	def interact(self, action_id):
		block_id, direction_id = self.decode_action(action_id)
		action_msg = self.message_protocol_kit.encode_action_from_pair(block_id, direction_id)
		self.connection.send_message(action_msg)
		state_code, reward, new_img, is_reset = self.receive_instruction_image
		return state_code, reward, new_img, is_reset

	def decode_action(self, action_id):
		if action_id == 80:
			return (5,20)
		else:
			block_id = action_id / self.num_direction
			direction_id = action_id % self.num_direction
			return (block_id, direction_id)

	def action2msg(self, action_id):
		if action_id == self.num_actions - 1:
			return "Stop"

		block_id = action_id / self.num_direction
		direction_id = action_id % self.num_direction

		if direction_id == 0:
			direction_id_str = "north"
		elif direction_id == 1:
			direction_id_str = "south"
		elif direction_id == 2:
			direction_id_str = "east"
		elif direction_id == 3:
			direction_id_str = "west"
		else:
			direction_id_str = None
			print "Error. Exiting"
			exit(0)
		return str(block_id) + " " + direction_id_str


	def build_inputs(self, img, instruction, previous_action):
		""" 
		img: collections.deque of numpy arrays
		instruction: list of word IDs
		previous_action: tuple of block_id and direction_id

		"""
		img_list = list(img)
		imgs = np.concatenate(img_list, axis=0)
		img_input = Variable(torch.from_numpy(imgs).float())
		img_input = img_input.unsqueeze(0)
		instruction_input = Variable(torch.from_numpy(np.array(instruction)))
		block = previous_action[1]
		direction = previous_action[0]
		block_input = Variable(torch.LongTensor([block])).unsqueeze(0)
		direction_input = Variable(torch.LongTensor([direction])).unsqueeze(0)
		action_input = (block_input, direction_input)
		return img_input, instruction_input, action_input

	def sample_policy(self, action_prob, method='random'):
		action_prob = action_prob.data.numpy().squeeze()
		print action_prob
		num_actions = len(action_prob)
		if method == 'random':
			action_id = np.random.choice(np.arange(num_actions), p=action_prob)
		elif method == 'greedy':
			action_id = np.argmax(action_prob)
		return action_id

	def train(self):

		for epoch in range(max_epochs):
			for sample_id in range(1, 1+train_size):
				state = collections.deque([], 5)
				init_imgs = self.model.image_encoder.build_init_images()
				state = collections.deque([],5)
				for img in init_imgs:
					state.append(img)
				_, _, img_state, instruction, trajectory = self.receive_instruction_image()
				img_state = np.transpose(img_state, (2,0,1))
				state.append(img_state)
				previous_action = self.null_previous_action
				instruction_ids = self.model.seq_encoder.instruction2id(instruction)
				inputs = self.build_inputs(state, instruction_ids, previous_action)

				traj_index = 0

				while True:
					action_id = trajectory[traj_index]
					action_msg = self.action2msg(action_id)
					# action_prob = self.model(inputs)
					# action_id = self.sample_policy(action_prob)
					# state_code, reward, new_env, is_reset = self.interact(action_id)
					self.connection.send_message(action_msg)
					(status_code, reward, new_env, is_reset) = self.receive_response()
					new_env = np.transpose(new_env, (2,0,1))
					if self.message_protocol_kit.is_reset_message(is_reset):
						self.connection.send_message("Ok-Reset")
						break

				return



if __name__ == '__main__':
	test = Inverse_agent()
	start = time.time()
	test.train()
	print time.time() - start
