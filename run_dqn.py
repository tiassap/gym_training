import argparse
import yaml
import gym
from utils.general import join
from utils.gym_wrapper import PreprocessingWrapper, preprocessing
from src.DeepQN import DQN
import warnings

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
			raise RuntimeError("{} method is {}. Method should be Policy Gradient".format(config_file, config["method"])) 

		print("Config file: {}".format(config_file))

		mode = "human" if not (args.train or args.record) else None
		env = gym.make(config["env"]["env_name"], render_mode = mode)
		env = PreprocessingWrapper(env, preprocessing)

		model = DQN(env, config)

		if args.train:
			print("Training")
			model.run_training()
			print("Finished training")

		elif args.record:
			pass

		else:
			pass

	else:
		# Show program documentation and usage
		with open('doc/run.txt') as f:
			doc = f.read()
		print(doc)