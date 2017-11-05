#!/usr/bin/python

import message_protocol_util as mpu
import reliable_connect as rc
import sys
from config import Config
import collections
import numpy as np
import time
import constants
from tqdm import tqdm
import random
import argparse
from policy_model import *
from critic_model import *

class Inverse_agent(object):
	"""inverse rl with GAN"""
	def __init__(self):		
		# logger.Log.open('./log.txt')

		# Connect to simulator
		# self.unity_ip = "0.0.0.0"
		self.unity_ip = "128.111.68.194"

		# self.PORT = 11000
		self.PORT = 46367

		# Size of image
		config = Config.parse("../../simulator2/Assets/config.txt")
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

		self.null_previous_direction = self.num_direction + 1

		self.gamma = 1.0

		self.policy_model = Policy_model(image_embed_dim=200, hidden_dim=250, action_dim_1=32, action_dim_2=24, inter_dim=120)
		self.critic_model = Critic_model(image_embed_dim=200, hidden_dim=250, direction_dim=24, inter_dim=120)

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

	def decode_action(self, action_id):
			block_id = action_id / self.num_direction
			direction_id = action_id % self.num_direction
			return (direction_id, block_id)

	def action2msg(self, block_id, direction_id):
		if direction_id == 4:
			return "Stop"

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

	def build_batch_inputs(self, trajectory, max_instruction=83):

		image_batch = [] # list of -1 * 15 * 120 * 120
		instruction_batch = [] # list of -1 * max_instruction
		lens_batch = [] # list of lengths
		previous_batch = [] # list of -1 * 1

		block_ids = []
		direction_ids = []

		for exp in trajectory:
			state = exp[0]
			block_id = exp[1]
			direction_id = exp[2]
			imgs = np.concatenate(list(state[0]), axis=0) 
			imgs = np.expand_dims(imgs, axis=0)
			image_batch.append(imgs)
			instruction_id = state[1]
			lens_batch.append(len(instruction_id))
			instruction_id_padded = np.lib.pad(instruction_id, (0, max_instruction - len(instruction_id)), 'constant', constant_values=(0,0))
			instruction_id_padded = np.expand_dims(instruction_id_padded, axis=0)
			instruction_batch.append(instruction_id_padded)
			previous_direction = state[2]
			previous_batch.append(previous_direction)
			block_ids.append(block_id)
			direction_ids.append(direction_id)

		image_batch = Variable(torch.from_numpy(np.concatenate(image_batch, axis=0)).float().cuda())
		instruction_batch = Variable(torch.LongTensor(np.concatenate(instruction_batch, axis=0)).cuda())
		previous_batch = Variable(torch.LongTensor(previous_batch).cuda().view(-1,1))
		block_ids = torch.LongTensor(block_ids).cuda()
		direction_ids = torch.LongTensor(direction_ids).cuda()

		return (image_batch, instruction_batch, lens_batch, previous_batch, block_ids, direction_ids)

	# def build_inputs(self, state):


	def exps_to_batchs(self, replay_memory, max_instruction=83):
		image_batch = []
		instruction_batch = []
		lens_batch = []
		previous_batch = []
		direction_batch = []
		block_batch = []
		for exp in replay_memory:
			gold_block_id = exp[1]
			gold_direction_id = exp[2]
			direction_batch.append(gold_direction_id)
			block_batch.append(gold_block_id)
			state = exp[0]
			imgs = np.concatenate(list(state[0]), axis=0)
			imgs = np.expand_dims(imgs, axis=0)
			image_batch.append(imgs)
			instruction_id = state[1]
			lens_batch.append(len(instruction_id))
			instruction_id_padded = np.lib.pad(instruction_id, (0, max_instruction - len(instruction_id)), 'constant', constant_values=(0,0))
			instruction_id_padded = np.expand_dims(instruction_id_padded, axis=0)
			instruction_batch.append(instruction_id_padded)
			previous_direction = state[2]
			previous_batch.append(previous_direction)

		image_batch = Variable(torch.from_numpy(np.concatenate(image_batch, axis=0)).float().cuda())
		instruction_batch = Variable(torch.LongTensor(np.concatenate(instruction_batch, axis=0)).cuda())
		previous_batch = Variable(torch.LongTensor(previous_batch).cuda().view(-1,1))
		block_batch = torch.LongTensor(block_batch).cuda()
		direction_batch = torch.LongTensor(direction_batch).cuda()

		return (image_batch, instruction_batch, lens_batch, previous_batch, block_batch, direction_batch)

	def sample_policy(self, action_prob, method='random'):
		action_prob = action_prob.data.cpu().numpy().squeeze()
		num_actions = len(action_prob)
		if method == 'random':
			action_id = np.random.choice(np.arange(num_actions), p=action_prob)
		elif method == 'greedy':
			action_id = np.argmax(action_prob)
		return action_id

	def test(self, saved_model, mode='dev', sample_method='greedy'):
		self.policy_model.load_state_dict(torch.load(saved_model))
		print 'Model reloaded'

		config = Config.parse("../../simulator2/Assets/config.txt")
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
			init_imgs = self.policy_model.image_encoder.build_init_images()
			for img in init_imgs:
				img_state.append(img)
			(status_code, bisk_metric, img, instruction, trajectory) = self.receive_instruction_image()

			gold_block_id = trajectory[0] / 4
			blocks_moved = []
			first = True
			
			img = np.transpose(img, (2,0,1))
			img_state.append(img)
			previous_direction = self.null_previous_direction
			instruction_ids = self.policy_model.seq_encoder.instruction2id(instruction)
			state = (img_state, instruction_ids, previous_direction)
			inputs = self.build_batch_inputs([(state, 0, 0)])
			_, block_prob, _ = self.policy_model(inputs)

			block_id_pred = self.sample_policy(block_prob.squeeze(), method='greedy')
			# block_id_pred = gold_block_id

			if block_id_pred == gold_block_id:
				print 'block correct'
				first_right += 1

			sum_bisk_metric += bisk_metric
			bisk_metrics.append(bisk_metric)

			action_paths = []

			while True:
				direction_prob, _, _ = self.policy_model(inputs)
				direction_id = self.sample_policy(direction_prob, method=sample_method)
				action_msg = self.action2msg(block_id_pred, direction_id)
				action_paths.append(action_msg)
				self.connection.send_message(action_msg)
				(_, reward, new_img, is_reset) = self.receive_response()

				new_img = np.transpose(new_img, (2,0,1))
				img_state.append(new_img)
				# previous_action = self.decode_action(action_id)
				previous_direction = direction_id
				state = (img_state, instruction_ids, previous_direction)
				inputs = self.build_batch_inputs([(state, 0, 0)])

				if self.message_protocol_kit.is_reset_message(is_reset):
					self.connection.send_message('Ok-Reset')
					break

			print 'action path:', action_paths
			print 'bisk metric until now:', np.mean(bisk_metrics)
			# print 'expert path:', expert_path

		avg_bisk_metric = sum_bisk_metric / float(test_size)
		median_bisk_metric = np.median(bisk_metrics)
		print "Avg. Bisk Metric " + str(avg_bisk_metric)
		print "Med. Bisk Metric " + str(median_bisk_metric)
		print "First Block accuracy  " + str(first_right/float(test_size))

if __name__ == '__main__':
	parser = argparse.ArgumentParser(description='Options at test stage')
	parser.add_argument('-model_path', default='models/rl_agent.pth', help='Path of saved model')
	parser.add_argument('-mode', default='dev', help='Test or Development')
	parser.add_argument('-sample_method', default='greedy')
	args = parser.parse_args()

	model_path = '../models/' + args.model_path

	agent = Inverse_agent()
	agent.policy_model.cuda()
	agent.test(model_path, mode=args.mode)



