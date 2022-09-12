import argparse
import gym
import torch
import numpy as np
import os
import sys

from src.policy_gradient import PolicyGradient
from utils.general import join, plot_combined

import matplotlib
matplotlib.use("agg")
import matplotlib.pyplot as plt

import random
import yaml

yaml.add_constructor("!join", join)

parser = argparse.ArgumentParser()
parser.add_argument("--config", required=False, type=str)
parser.add_argument("--train", action='store_true')
parser.add_argument("--record", action='store_true')

if __name__ == "__main__":
	args = parser.parse_args()

	if args.config is not None:
		config_file = open("config/{}.yml".format(args.config))
		config = yaml.load(config_file, Loader=yaml.FullLoader)
		print("Config file: {}".format(config_file))

		mode = "human" if not (args.train or args.record) else None
		env = gym.make(config["env"]["env_name"], render_mode = mode)

		# if config["env"]["method"] == "Policy Gradient" :
		# 	print("Policy Gradient method")
		# 	model = PolicyGradient(env, config)

		model = PolicyGradient(env, config)

		if args.train:
			print("Training")
			model.policy.load_state_dict(torch.load(
				config["output"]["model_output"],
				map_location="cpu"
			))
			model.run_training()
			print("Finished training")

		elif args.record:
			pass

		else:
			print("Running simulation")
			# load_state_dict
			model.policy.load_state_dict(torch.load(
				config["output"]["model_output"],
				map_location="cpu"
			))

			model.run_simulation()

	else:
		# Show program documentation and usage
		with open('doc/run.txt') as f:
			doc = f.read()
		print(doc)