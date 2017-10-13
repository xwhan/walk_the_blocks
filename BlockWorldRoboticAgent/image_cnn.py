#!/usr/bin/python 

import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F

import numpy as np

class CNN_encoder(nn.Module):
	"""docstring for CNN_encoder"""
	def __init__(self, input_channels, output_size, image_dim):
		super(CNN_encoder, self).__init__()
		self.input_channels = input_channels
		self.output_size = output_size
		self.image_dim = image_dim

		self.conv1 = nn.Conv2d(input_channels, 32, kernel_size=8, stride=4, padding=2)
		self.conv2 = nn.Conv2d(32, 32, kernel_size=8, stride=4, padding=3)
		self.conv3 = nn.Conv2d(32, 32, kernel_size=4, stride=2, padding=1)
		self.affine = nn.Linear(512, self.output_size)

	def forward(self, image):
		x = F.relu(self.conv1(image))
		x = F.relu(self.conv2(x))
		x = F.relu(self.conv3(x))
		x = self.affine(x.view(-1,512))
		return x

	def normalize(self, image):
		pass

	def build_init_images(self):
		return [np.zeros((3,120,120))] * 4

if __name__ == '__main__':
	test_input = Variable(torch.randn(10,15,120,120))
	encoder = CNN_encoder(15, 200, 120)
	out = encoder(test_input)
	print out.size()