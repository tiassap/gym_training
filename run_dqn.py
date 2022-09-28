import torch
import argparse
import yaml
import gym
from utils.general import join, get_pretrained_model
from utils.gym_wrapper import PreprocessingWrapper, preprocessing
from src.DeepQN import DQN
import warnings
from datetime import datetime
import os

warnings.filterwarnings("ignore", module=r"gym")


parser = argparse.ArgumentParser()
parser.add_argument("--config", required=False, type=str)
parser.add_argument("--train", action='store_true')
parser.add_argument("--record", action='store_true')

if __name__== "__main__":
	args = parser.parse_args()

	if args.config is not None:
		yaml.add_constructor("!join", join)
		config_file = open("config/{}.yml".format(args.config))
		config = yaml.load(config_file, Loader=yaml.FullLoader)

		if config["method"] != "DQN":
			raise RuntimeError("Method in config file {} is {}. Method should be Double Q-Network".format(config_file, config["method"])) 

		print("Config file: {}".format(config_file))

		if not os.path.exists(config["output"]["model_output"]):
			os.makedirs(config["output"]["model_output"])
		
		if not os.path.exists(config["output"]["record_path"]):
			os.makedirs(config["output"]["record_path"])

		if not os.path.exists(config["model_training"]["load_path"]):
			os.makedirs(config["model_training"]["load_path"])

		mode = "human" if not (args.train or args.record) else None
		env = gym.make(config["env"]["env_name"], full_action_space=False, render_mode = mode)
		env = PreprocessingWrapper(env, preprocessing)
		if args.record:
			env = gym.wrappers.RecordVideo(env, 
			config["output"]["record_path"], 
			step_trigger=lambda x: x % 10 == 0,
			name_prefix=config["env"]["env_name"])

		model = DQN(env, config)

		if config["env"]["use_pretrained_weights"]:
			path = get_pretrained_model(config["model_training"]["load_path"])[0]
			# path = "/home/tias/Data_science/1_project/gym_training/output/Pong-v4/models/model_20220926151732_250000.weights.pt"
			model.Q.load_state_dict(torch.load(path, map_location="cpu"))
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