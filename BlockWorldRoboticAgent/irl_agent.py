#!/usr/bin/python

import message_protocol_util as mpu
import reliable_connect as rc
import sys
import logger
from config import Config
import collections
import numpy as np
import time
import constants
from tqdm import tqdm
import random
import argparse

from hyperparas import *
from attention import *

class Inverse_agent(object):
	"""inverse rl with GAN"""
	def __init__(self):		
		logger.Log.open('./log.txt')

		# Connect to simulator
		# if len(sys.argv) < 2:
		# 	logger.Log.info("IP not given. Using localhost i.e. 0.0.0.0")
		# 	self.unity_ip = "0.0.0.0"
		# else:
		# 	self.unity_ip = sys.argv[1]
		self.unity_ip = "0.0.0.0"

		# if len(sys.argv) < 3:
		# 	logger.Log.info("PORT not given. Using 11000")
		# 	self.PORT = 11000
		# else:
		# 	self.PORT = int(sys.argv[2])
		self.PORT = 11000

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

		self.gamma = 1.0

		self.model = Context_attention(image_embed_dim=200, hidden_dim=250, action_dim_1=32, action_dim_2=24, inter_dim=120)

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
			block_id = action_id / self.num_direction
			direction_id = action_id % self.num_direction
			return (direction_id, block_id)

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
		img_input = Variable(torch.from_numpy(imgs).float().cuda())
		img_input = img_input.unsqueeze(0)
		instruction_input = Variable(torch.from_numpy(np.array(instruction)).cuda())
		block = previous_action[1]
		direction = previous_action[0]
		block_input = Variable(torch.LongTensor([block]).cuda()).unsqueeze(0)
		direction_input = Variable(torch.LongTensor([direction]).cuda()).unsqueeze(0)
		action_input = (block_input, direction_input)
		return img_input, instruction_input, action_input

	def sample_policy(self, action_prob, method='random'):
		action_prob = action_prob.data.cpu().numpy().squeeze()
		num_actions = len(action_prob)
		if method == 'random':
			action_id = np.random.choice(np.arange(num_actions), p=action_prob)
		elif method == 'greedy':
			action_id = np.argmax(action_prob)
		return action_id

	def test(self, saved_model, cuda=True, mode='dev'):
		if cuda:
			self.model.cuda()
		self.model.load_state_dict(torch.load(saved_model))
		print 'Model reloaded'

		config = Config.parse("../BlockWorldSimulator/Assets/config.txt")
		constants_hyperparam = constants.constants
		if mode == 'dev':
			assert config.data_mode == Config.DEV
			test_size = constants_hyperparam["dev_size"]
		elif mode == 'test':
			assert config.data_mode == Config.TEST 
			test_size = constants_hyperparam['test_size']
		else:
			assert config.data_mode == Config.TRAIN 
			test_size = 1000

		sum_bisk_metric = 0
		bisk_metrics = []
		sum_reward = 0
		sum_steps = 0
		right_block = 0
		first_right = 0

		for sample_id in tqdm(range(test_size)):
			img_state = collections.deque([], 5)
			init_imgs = self.model.image_encoder.build_init_images()
			for img in init_imgs:
				img_state.append(img)
			(status_code, bisk_metric, img, instruction, trajectory) = self.receive_instruction_image()

			gold_block_id = int(trajectory[0] / 4.0)
			blocks_moved = []
			first = True
			first_right = 0

			img = np.transpose(img, (2,0,1))
			img_state.append(img)
			previous_action = self.null_previous_action
			instruction_ids = self.model.seq_encoder.instruction2id(instruction)
			inputs = self.build_inputs(img_state, instruction_ids, previous_action)

			sum_bisk_metric += bisk_metric
			bisk_metrics.append(bisk_metric)

			while True:
				action_prob = self.model(inputs).squeeze()
				action_id = self.sample_policy(action_prob, method='greedy')
				
				block_id = action_id / self.num_direction
				if first:
					first = False
					if block_id == gold_block_id:
						first_right += 1

				action_msg = self.action2msg(action_id)
				self.connection.send_message(action_msg)
				(_, reward, new_img, is_reset) = self.receive_response()

				new_img = np.transpose(new_img, (2,0,1))
				img_state.append(new_img)
				previous_action = self.decode_action(action_id)
				inputs = self.build_inputs(img_state, instruction_ids, previous_action)

				if self.message_protocol_kit.is_reset_message(is_reset):
					self.connection.send_message('Ok-Reset')
					break

		avg_bisk_metric = sum_bisk_metric / float(test_size)
		median_bisk_metric = np.median(bisk_metrics)
		print "Avg. Bisk Metric " + str(avg_bisk_metric)
		print "Med. Bisk Metric " + str(median_bisk_metric)
		print "First Block accuracy  " + str(first_right/float(test_size))

if __name__ == '__main__':
	parser = argparse.ArgumentParser(description='Options at test stage')
	parser.add_argument('-model_path', default='models/rl_agent.pth', help='Path of saved model')
	parser.add_argument('-mode', default='dev', help='Test or Development')
	args = parser.parse_args()

	model_path = 'models/' + args.model_path

	agent = Inverse_agent()
	agent.model.cuda()
	agent.test(model_path, mode=args.mode)



