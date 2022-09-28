import numpy as np
import torch
import torch.nn as nn
from utils.network_utils import device, np2torch, batch_iterator
import gym
# import os
from utils.general import get_logger, Progbar, export_plot
from src.nnetwork import mlp
from src.policy import CategoricalPolicy, GaussianPolicy
from datetime import datetime


class BaselineNetwork(nn.Module):

	def __init__(self, env, config):
		super().__init__()
		self.config = config
		self.env = env
		self.lr = self.config["hyper_params"]["learning_rate"]
		self.network = mlp(
			env.observation_space.shape[0], 1, config["hyper_params"]["n_layers"], config["hyper_params"]["layer_size"])
		self.optimizer = torch.optim.Adam(
			self.network.parameters(), lr=self.lr)

	def forward(self, observations):
		output = self.network(observations)
		output = output.squeeze()
		return output

	def calculate_advantage(self, returns, observations):
		"""
		Returns:
			advantages (np.array): returns - baseline values  (shape [batch size])
		"""
		observations = np2torch(observations)
		advantages = returns - \
			self.forward(observations).detach().cpu().numpy()
		return advantages

	def update_baseline(self, returns, observations):
		"""
		Performs back propagation to update the weights of the baseline network according to MSE loss

		Args:
			returns (np.array): the history of discounted future returns for each step (shape [batch size])
			observations (np.array): observations at each step (shape [batch size, dim(observation space)])
			called batch_iterator (implemented in utils/network_utils.py).
		"""
		returns = np2torch(returns)
		observations = np2torch(observations)

		criterion = torch.nn.MSELoss()
		# import pdb
		# pdb.set_trace()
		for obs, ret in batch_iterator(observations, returns, batch_size=100, shuffle=True):
			self.optimizer.zero_grad()
			loss = criterion(self.forward(obs), ret)
			loss.backward()
			self.optimizer.step()

			if loss <= 0.05:
				break


class PolicyGradient(object):

	def __init__(self, env, config, seed=None, logger=None):

		# if not os.path.exists(config["output"]["model_output"]):
		# 	os.makedirs(config["output"]["model_output"])
		
		# if not os.path.exists(config["model_training"]["load_path"]):
		# 	os.makedirs(config["model_training"]["load_path"])

		self.config = config
		self.seed = seed

		self.logger = logger
		if logger is None:
			# self.logger = get_logger(config["output"]["log_path"].format(seed))
			self.logger = get_logger(config["output"]["log_path"])

		self.env = env
		self.env.reset(seed=self.seed)

		# discrete vs continuous action space
		self.discrete = isinstance(env.action_space, gym.spaces.Discrete)
		self.observation_dim = self.env.observation_space.shape[0]
		self.action_dim = (
			self.env.action_space.n if self.discrete else self.env.action_space.shape[0]
		)

		self.lr = self.config["hyper_params"]["learning_rate"]

		self.init_policy()

		if config["model_training"]["use_baseline"]:
			self.baseline_network = BaselineNetwork(env, config).to(
				torch.device("cuda" if torch.cuda.is_available() else "cpu")
			)

	def init_policy(self):
		network = mlp(
			self.observation_dim,
			self.action_dim,
			self.config["hyper_params"]["n_layers"],
			self.config["hyper_params"]["layer_size"]
		)
		network = network.to(device)

		if self.discrete:
			self.policy = CategoricalPolicy(network)
		else:
			self.policy = GaussianPolicy(network, self.action_dim)

		self.optimizer = torch.optim.Adam(self.policy.parameters(), lr=self.lr)

	def init_averages(self):
		self.avg_reward = 0.0
		self.max_reward = 0.0
		self.std_reward = 0.0
		self.eval_reward = 0.0

	def update_averages(self, rewards, scores_eval):
		self.avg_reward = np.mean(rewards)
		self.max_reward = np.max(rewards)
		self.std_reward = np.sqrt(np.var(rewards) / len(rewards))

		if len(scores_eval) > 0:
			self.eval_reward = scores_eval[-1]

	def record_summary(self, t):
		pass

	def sample_path(self, env, num_episodes=None):
		"""
		Sample paths (trajectories) from the environment.

		Args:
			num_episodes (int): the number of episodes to be sampled
				if none, sample one batch (size indicated by config file)
			env (): open AI Gym envinronment

		Returns:
			paths (list): a list of paths. Each path in paths is a dictionary with
						path["observation"] a numpy array of ordered observations in the path
						path["actions"] a numpy array of the corresponding actions in the path
						path["reward"] a numpy array of the corresponding rewards in the path
			total_rewards (list): the sum of all rewards encountered during this "path"

		"""
		episode = 0
		episode_rewards = []
		paths = []
		t = 0

		while num_episodes or t < self.config["hyper_params"]["batch_size"]:
			state = env.reset()
			states, actions, rewards = [], [], []
			episode_reward = 0

			for step in range(self.config["hyper_params"]["max_ep_len"]):
				states.append(state)
				action = self.policy.act(states[-1][None])[0]
				state, reward, done, info = env.step(action)
				actions.append(action)
				rewards.append(reward)
				episode_reward += reward
				t += 1
				if done or step == self.config["hyper_params"]["max_ep_len"] - 1:
					episode_rewards.append(episode_reward)
					break
				if (not num_episodes) and t == self.config["hyper_params"][
					"batch_size"
				]:
					break

			path = {
				"observation": np.array(states),
				"reward": np.array(rewards),
				"action": np.array(actions),
			}
			paths.append(path)
			episode += 1
			if num_episodes and episode >= num_episodes:
				break

		return paths, episode_rewards

	def get_returns(self, paths):
		all_returns = []

		for path in paths:
			rewards = path["reward"]
			returns = []

			# Loop for each step t in [1,T]. Starting from T
			for t in reversed(range(len(rewards))):
				if t == len(rewards) - 1:
					# G_t = R_t
					returns += [rewards[t]]
				else:
					# G_t = R_t + γ G_{t+1}
					returns += [rewards[t] + self.config["hyper_params"]
								["gamma"] * returns[-1]]

			# Reverse list 'returns'
			returns.reverse()
			all_returns.append(returns)
		returns = np.concatenate(all_returns)

		return returns

	def calculate_advantage(self, returns, observations):
		"""
		Calculates the advantage for each of the observations

		Args:
			returns (np.array): shape [batch size]
			observations (np.array): shape [batch size, dim(observation space)]

		Returns:
			advantages (np.array): shape [batch size]
		"""
		if self.config["model_training"]["use_baseline"]:
			# override the behavior of advantage by subtracting baseline
			advantages = self.baseline_network.calculate_advantage(
				returns, observations
			)
		else:
			advantages = returns

		if self.config["model_training"]["normalize_advantage"]:
			advantages = (
				advantages - advantages.mean()) / advantages.std()

		return advantages

	def update_policy(self, observations, actions, advantages):
		"""
		Args:
			observations (np.array): shape [batch size, dim(observation space)]
			actions (np.array): shape [batch size, dim(action space)] if continuous
								[batch size] (and integer type) if discrete
			advantages (np.array): shape [batch size]
	"""
		observations = np2torch(observations)
		actions = np2torch(actions)
		advantages = np2torch(advantages)
		self.optimizer.zero_grad()
		probability = self.policy.action_distribution(observations)
		policy_gradient = (probability.log_prob(actions) * advantages).sum() * -1
		policy_gradient.backward()
		self.optimizer.step()

	def train(self):
		last_record = 0

		self.init_averages()
		all_total_rewards = ([])  
		# averaged_total_rewards = []  # the returns for each iteration
		averaged_total_rewards = np.array([])

		# set policy to device
		self.policy = self.policy.to(
			torch.device("cuda" if torch.cuda.is_available() else "cpu")
		)
		t = 0
		while True:
			t += 1
			# collect a minibatch of samples
			paths, total_rewards = self.sample_path(self.env)
			all_total_rewards.extend(total_rewards)
			observations = np.concatenate(
				[path["observation"] for path in paths])
			actions = np.concatenate([path["action"] for path in paths])
			rewards = np.concatenate([path["reward"] for path in paths])
			# compute Q-val estimates (discounted future returns) for each time step
			returns = self.get_returns(paths)

			# advantage will depend on the baseline implementation
			advantages = self.calculate_advantage(returns, observations)

			# run training operations
			if self.config["model_training"]["use_baseline"]:
				self.baseline_network.update_baseline(returns, observations)
			self.update_policy(observations, actions, advantages)

			# logging
			if t % self.config["model_training"]["summary_freq"] == 0:
				self.update_averages(total_rewards, all_total_rewards)
				self.record_summary(t)

			# compute reward statistics for this batch and log
			avg_reward = np.mean(total_rewards)
			sigma_reward = np.sqrt(np.var(total_rewards) / len(total_rewards))
			msg = "Average reward: {:04.2f} +/- {:04.2f}".format(
				avg_reward, sigma_reward
			)
			# averaged_total_rewards.append(avg_reward)
			averaged_total_rewards = np.append(averaged_total_rewards, avg_reward)

			self.logger.info(msg)

			if t % self.config["model_training"]["saving_freq"] == 0:
				self.save_model(self.config["output"]["model_output"], t)

			if t >= self.config["hyper_params"]["num_batches"]:
				break
			
			if averaged_total_rewards[-10:].mean() >= self.config["env"]["min_expected_reward"]:
				break

		self.logger.info("- Training done.")

		self.save_model(self.config["model_training"]["load_path"], t)
		

	def evaluate(self, env=None, num_episodes=1, logging=False):
		"""
		Evaluates the return for num_episodes episodes.
		Not used right now, all evaluation statistics are computed during training
		episodes.
		"""
		if env == None:
			env = self.env
		paths, rewards = self.sample_path(env, num_episodes)
		avg_reward = np.mean(rewards)
		sigma_reward = np.sqrt(np.var(rewards) / len(rewards))
		if logging:
			msg = "Average reward: {:04.2f} +/- {:04.2f}".format(
				avg_reward, sigma_reward
			)
			self.logger.info(msg)
		return avg_reward


	def record(self):
		"""
		Recreate an env and record a video for one episode
		"""
		env = gym.make(self.config["env"]["env_name"])
		env.reset(seed=self.seed)
		env = gym.wrappers.RecordVideo(
			env,
			# self.config["output"]["record_path"].format(self.seed),
			self.config["output"]["record_path"],
			step_trigger=lambda x: x % 100 == 0,
		)
		self.evaluate(env, 1)


	def save_model(self, path, t):
		FORMAT = '%Y%m%d%H%M%S'
		datenow = datetime.now().strftime(FORMAT)

		# Pathname format: "output_path/model_{datetime}_{t}.weights"
		PATH = path + "/model_{}_{}.weights.pt".format(datenow, t)
		print("		saving ", PATH)
		torch.save(self.policy.state_dict(), PATH)

	def run_training(self):
		"""
		Apply procedures of training for a PG.
		"""
		# record one game at the beginning
		if self.config["env"]["record"]:
			self.record()
		# model
		self.train()
		# record one game at the end
		if self.config["env"]["record"]:
			self.record()

	def run_simulation(self):
		# for t in range(self.config["hyper_params"]["num_batches"]):
		paths, rewards = self.sample_path(self.env, 3)
		avg_reward = np.mean(rewards)
		sigma_reward = np.sqrt(np.var(rewards) / len(rewards))
		msg = "Average reward: {:04.2f} +/- {:04.2f}".format(
				avg_reward, sigma_reward
			)
		print(msg)


