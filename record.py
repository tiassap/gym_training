import argparse
import os
import sys
import numpy as np
import torch
import gym
import matplotlib

matplotlib.use("agg")
import matplotlib.pyplot as plt
import unittest
from utils.general import join, plot_combined
from src.policy_gradient import PolicyGradient

import random
import yaml

yaml.add_constructor("!join", join)

parser = argparse.ArgumentParser()
parser.add_argument("--config_filename", required=False, type=str)



if __name__ == "__main__":
	args = parser.parse_args()

	if args.config_filename is not None:
		config_file = open("config/{}.yml".format(args.config_filename))
		config = yaml.load(config_file, Loader=yaml.FullLoader)

		env = gym.make(config["env"]["env_name"])

		model = PolicyGradient(env, config)

		model.policy.load_state_dict(torch.load(
			config["output"]["model_output"],
			map_location="cpu"
		))

		# Create video
		model.record()

	else:
		print("Please specify config file name by --config_filename=<file_name>")