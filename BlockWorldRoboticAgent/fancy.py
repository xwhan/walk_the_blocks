#!/usr/bin/python

import message_protocol_util as mpu
import reliable_connect as rc
import sys
import logger
from config import Config

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
		self.config = Config
		image_dim = self.config.screen_size
		print image_dim

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

		

if __name__ == '__main__':
	test = Inverse_agent()

