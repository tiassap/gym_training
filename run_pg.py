import argparse
import gym
import torch
import os
from src.policy_gradient import PolicyGradient
from utils.general import join, plot_combined, get_pretrained_model
from datetime import datetime
import warnings
import yaml

warnings.filterwarnings("ignore", module=r"gym")



parser = argparse.ArgumentParser()
parser.add_argument("--config", required=False, type=str)
parser.add_argument("--train", action='store_true')
parser.add_argument("--record", action='store_true')

if __name__ == "__main__":
	args = parser.parse_args()

	if args.config is not None:
		yaml.add_constructor("!join", join)
		config_file = open("config/{}.yml".format(args.config))
		config = yaml.load(config_file, Loader=yaml.FullLoader)

		if config["method"] != "Policy Gradient":
			raise RuntimeError("{} method is {}. Method should be Policy Gradient".format(config_file, config["method"])) 

		print("Config file: {}".format(config_file))

		if not os.path.exists(config["output"]["model_output"]):
			os.makedirs(config["output"]["model_output"])
		
		if not os.path.exists(config["output"]["record_path"]):
			os.makedirs(config["output"]["record_path"]+"450")

		if not os.path.exists(config["model_training"]["load_path"]):
			os.makedirs(config["model_training"]["load_path"])

		mode = "human" if not (args.train or args.record) else None
		env = gym.make(config["env"]["env_name"], render_mode = mode)
		if args.record:
			env = gym.wrappers.RecordVideo(env, 
			config["output"]["record_path"]+"450", 
			step_trigger=lambda x: x % 10 == 0,
			name_prefix=config["env"]["env_name"])

		model = PolicyGradient(env, config, seed=1)

		if config["env"]["use_pretrained_weights"]:
			# path = get_pretrained_model(config["model_training"]["load_path"])[0]
			path = '/home/tias/Data_science/1_project/gym_training/output/HalfCheetah-v4/models/model_20220926115422_150.weights.pt'
			model.policy.load_state_dict(torch.load(path, map_location="cpu"))
			print("Pretrained weights loading successful from path: {}".format(path))

		if args.train:
			print("Training start... with {} | {}".format(("GPU" if torch.cuda.is_available() else "CPU"), datetime.now().strftime('%Y-%m-%d|%H:%M:%S')))
			print("Num of actions: {}".format(env.action_space.n))
			model.run_training()
			print("Finished training | {}".format(datetime.now().strftime('%Y-%m-%d|%H:%M:%S')))

		else:
			assert config["env"]["use_pretrained_weights"], "Please set the config 'use_pretrained_weights' to be True for recording and simulation."
			if args.record:
				print("Recording simulation using weight: {}".format(path))
			else:
				print("Running simulation using weight: {}".format(path))
			model.run_simulation()

	else:
		# Show program documentation and usage
		with open('doc/run.txt') as f:
			doc = f.read()
		print(doc)