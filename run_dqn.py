import argparse
import yaml
import gym

parser = argparse.ArgumentParser()
parser.add_argument("--config", required=False, type=str)
parser.add_argument("--train", action='store_true')
parser.add_argument("--record", action='store_true')

if __name__== "__main__":
	args = parser.parse_args()

	if args.config is not None:
		config_file = open("config/{}.yml".format(args.config))
		config = yaml.load(config_file, Loader=yaml.FullLoader)

		if config["method"] != "DQN":
			raise RuntimeError("{} method is {}. Method should be Policy Gradient".format(config_file, config["method"])) 

		print("Config file: {}".format(config_file))

		mode = "human" if not (args.train or args.record) else None
		env = gym.make(config["env"]["env_name"], render_mode = mode)

		model = ###

		if args.train:
			print("Training")
			if input("Use pre-trained model? (y/n): ") == "y":
				model.policy.load_state_dict(torch.load(
				config["output"]["model_output"],
				map_location="cpu"
				))

			model.run_training()
			print("Finished training")

		elif args.record:
			model.policy.load_state_dict(torch.load(
			config["output"]["model_output"],
			map_location="cpu"
		))
			# Create video
			model.record()
			print("Video created {}".format(config["output"]["model_output"]))

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