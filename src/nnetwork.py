import torch
import torch.nn as nn


def mlp(input_size, output_size, n_layers, size):
	"""
	Multi Layer Perceptron network
	"""
	modules = nn.ModuleList([])
	modules.append(nn.Linear(input_size, size))
	modules.append(nn.ReLU())  
	for _ in range(n_layers - 1):
		modules.append(nn.Linear(size, size))
		modules.append(nn.ReLU())
	modules.append(nn.Linear(size, output_size))
	model = nn.Sequential(*modules)
	
	return model

def conv_1():
	pass
