if __name__ == "__main__":
	import sys
	sys.path.append('/home/tias/Data_science/1_project/gym_training')

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import copy
import numpy as np

import gym
from utils.replay_buffer import ReplayBuffer, sample_n_unique
from utils.gym_wrapper import PreprocessingWrapper, preprocessing
from torch.utils.tensorboard import SummaryWriter


class NN_VFA(nn.Module):
	def __init__(self, env, config):
		super().__init__()
		state_shape = list(env.observation_space.shape)
		img_height, img_width, n_channels = state_shape
		num_actions = env.action_space.n
		history = config["hyper_params"]["state_history"]

		self.network = nn.Sequential(
			nn.Conv2d(n_channels * history, 32, (8, 8), stride=4, padding=2),
			nn.ReLU(),
			nn.Conv2d(32, 64, (4, 4), stride=2, padding=0),
			nn.ReLU(),
			nn.Conv2d(64, 64, (3, 3), stride=1, padding=0),
			nn.ReLU(),
			nn.Flatten(),
			nn.Linear(3136, 512),
			nn.ReLU(),
			nn.Linear(512, num_actions)
		)

	def forward(self, x):
		return self.network(x)


class DQN(object):
	def __init__(self, env, config) -> None:

		if not os.path.exists(config["output"]["output_path"]):
			os.makedirs(config["output"]["output_path"])

		# Initialize model Q and Q_target
		self.Q = NN_VFA(env, config)
		self.Q_target = copy.deepcopy(self.Q)
		self.device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

		self.Q = self.Q.to(self.device)
		self.Q_target = self.Q_target.to(self.device)
		self.optimizer = torch.optim.Adam(self.Q.parameters())
		self.lr = config["hyper_params"]["lr_begin"]
		self.epsilon = config["hyper_params"]["eps_begin"]

		self.env = env
		self.config = config

		self.summary_writer = SummaryWriter(
			self.config["output"]["tensorboard_output"], max_queue=1e6)
		# self.avg_scores = []
		# self.std_scores = []
		# self.t_eval = []


	def normalization(self, state):
		return (state / 255).float()


	def play(self, train=False, eval_episode=30):
		buffer_size = self.config["hyper_params"]["buffer_size"]
		history = self.config["hyper_params"]["state_history"]
		n_step = self.config["hyper_params"]["nsteps_train"]
		replay_buffer = ReplayBuffer(buffer_size, history)

		t = 0
		episode = 0
		rewards = []
		max_q_vals = []


		while t <= n_step:
			state = self.env.reset()
			episode_reward = 0
			episode_q_vals = 0

			while True:
				t += 1

				idx = replay_buffer.store_frame(state)

				# Concatenate 4 frames of recent observation. Add padding 0 if necessary.
				q_input = replay_buffer.encode_recent_observation()

				best_action, q_vals = self.get_best_action(q_input)

				if train:
					# Use e-greedy exploration strategy
					self.update_epsilon(t)
					action = self.epsilon_greedy(best_action)
				else:
					action = best_action

				# action = self.env.action_space.sample()

				new_state, reward, done, info = self.env.step(action)

				replay_buffer.store_effect(idx, action, reward, done)
				state = new_state

				episode_reward += reward
				episode_q_vals += np.max(q_vals)  # Max Q over a

				if (train and t > self.config["hyper_params"]["learning_start"]):
					if t % self.config["hyper_params"]["learning_freq"] == 0:
						self.train_step(replay_buffer, t)

					if t % self.config["hyper_params"]["target_update_freq"] == 0:
						# Update target parameter
						self.Q_target.load_state_dict(self.Q.state_dict())

					if t % self.config["model_training"]["eval_freq"] == 0:
						scores = self.play(train=False)
						# scores : (rewards, max_q_vals)
						self.evaluate(scores, t)

					if t % self.config["model_training"]["saving_freq"] == 0:
						self.save_model(t)

				if done or t >= n_step:
					episode += 1
					rewards += [episode_reward]
					max_q_vals += [episode_q_vals]
					self.tf_add_summary(episode_reward, episode_q_vals, episode)
					break

			# Return 'scores' for evaluation
			if not train and (episode >= eval_episode or t >= n_step):
				return (rewards, max_q_vals)


	def epsilon_greedy(self, action):
		# epsilon = self.config["model_training"]["soft_epsilon"]
		if np.random.random() < self.epsilon:
			return self.env.action_space.sample()
		else:
			return action


	def get_best_action(self, state: torch.Tensor):
		"""
		Return best action

		Args:
				state: 4 consecutive observations from gym
		Returns:
				action: (int)
				action_values: (np array) q values for all actions
		"""
		with torch.no_grad():
			s = torch.tensor(state, dtype=torch.uint8,
							 device=self.device).unsqueeze(0)
			s = self.normalization(s)
			action_values = self.get_q_values(
				s, 'q_network').squeeze().to('cpu').tolist()
		action = np.argmax(action_values)
		return action, action_values


	def get_q_values(self, state, network):
		out = None

		# Swap the order of image element to n, C, H, W
		input = torch.permute(state, (0, 3, 1, 2))

		if network == 'q_network':
			out = self.Q(input)
		elif network == 'target_network':
			out = self.Q_target(input)

		return out


	def train_step(self, replay_buffer, t):
		states, actions, rewards, next_states, dones = replay_buffer.sample(
			batch_size=32)

		states = torch.tensor(states, dtype=torch.uint8, device=self.device)
		actions = torch.tensor(actions, dtype=torch.uint8, device=self.device)
		rewards = torch.tensor(rewards, dtype=torch.float, device=self.device)
		next_states = torch.tensor(
			next_states, dtype=torch.uint8, device=self.device)
		dones = torch.tensor(dones, dtype=torch.bool, device=self.device)

		self.optimizer.zero_grad()

		q_values = self.get_q_values(self.normalization(states), 'q_network')

		with torch.no_grad():
			target_q_values = self.get_q_values(
				self.normalization(next_states), 'target_network')

		loss = self.calc_loss(q_values, target_q_values,
							  actions, rewards, dones)

		loss.backward()

		total_norm = torch.nn.utils.clip_grad_norm_(
			self.Q.parameters(), self.config["model_training"]["clip_val"])

		self.update_lr(t)
		for group in self.optimizer.param_groups:
			group['lr'] = self.lr

		self.optimizer.step()

		return loss.item(), total_norm.item()


	def update_lr(self, t):
		lr_begin = self.config["hyper_params"]["lr_begin"]
		lr_end = self.config["hyper_params"]["lr_end"]
		n_steps = self.config["hyper_params"]["lr_nsteps"]

		self.lr = self.linear_schedule(t, lr_begin, lr_end, n_steps)


	def update_epsilon(self, t):
		eps_begin = self.config["hyper_params"]["eps_begin"]
		eps_end = self.config["hyper_params"]["eps_end"]
		n_steps = self.config["hyper_params"]["eps_nsteps"]

		self.epsilon = self.linear_schedule(t, eps_begin, eps_end, n_steps)


	def linear_schedule(self, t, val_begin, val_end, n_steps):
		if t <= n_steps:
			return val_begin + (val_end - val_begin) / n_steps * t
		else:
			return val_end


	def save_model(self, t):
		from datetime import datetime
		FORMAT = '%Y%m%d%H%M%S'
		datenow = datetime.now().strftime(FORMAT)

		# Pathname format: "output_path/model_timestep-{t}_{datetime}.weights"
		PATH = self.config["output"]["model_output"].format(t, datenow)
		torch.save(self.Q.state_dict(), PATH)


	def calc_loss(self, q_values: torch.Tensor, target_q_values: torch.Tensor,
				  actions: torch.Tensor, rewards: torch.Tensor, done_mask: torch.Tensor):

		num_actions = self.env.action_space.n
		gamma = self.config["hyper_params"]["gamma"]
		q_ = (F.one_hot(actions.to(int), num_classes=num_actions)
			  * q_values).sum(dim=1)
		q_target = torch.where(
			done_mask, rewards, rewards + gamma * torch.max(target_q_values, axis=1)[0])
		loss = F.mse_loss(q_, q_target, reduction='mean')

		return loss


	def evaluate(self, scores, t):
		"""
		Evaluate best action result from trained network.
		"""
		avg_reward = np.mean(scores[0])
		max_reward = np.max(scores[1])
		std_reward = np.std(scores[0])

		avg_q = np.mean(scores[1])
		max_q = np.max(scores[1])
		std_q = np.std(scores[1])

		total_step = self.config["hyper_params"]["nsteps_train"]
		print("Training step {}/{} \t : Score {} +/-{}".format(t,
			  total_step, avg_reward, std_reward))
		# self.avg_scores.append(avg_score)
		# self.std_scores.append(std_score)
		# self.t_eval.append(t)
		# self.summary_writer.add_scalar('loss', loss, t)
		self.summary_writer.add_scalar('Avg_Reward', avg_reward, t)
		self.summary_writer.add_scalar('Max_Reward', max_reward, t)
		self.summary_writer.add_scalar('Std_Reward', std_reward, t)
		self.summary_writer.add_scalar('Avg_Q', avg_q, t)
		self.summary_writer.add_scalar('Max_Q', max_q, t)
		self.summary_writer.add_scalar('Std_Q', std_q, t)


	def tf_add_summary(self, reward, Q_val, episode):
		"""Reward and Q values during training"""
		self.summary_writer.add_scalar('Reward @training', reward, episode)
		self.summary_writer.add_scalar('Q values @training', Q_val, episode)


	def run_training(self):
		self.play(train=True)


	def run_simulation(self):
		self.play(eval_episode=20)


if __name__ == "__main__":
	# For debugging
	import warnings
	import yaml
	from utils.general import join

	warnings.filterwarnings("ignore", module=r"gym")
	yaml.add_constructor("!join", join)

	env = gym.make("ALE/Breakout-v5", render_mode="human")
	env = PreprocessingWrapper(env, preprocessing)

	config_file = open(
		"/home/tias/Data_science/1_project/gym_training/config/dummy-breakout_test.yml")
	config = yaml.load(config_file, Loader=yaml.FullLoader)
	model = DQN(env, config)
	model.run_simulation()
