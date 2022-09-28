if __name__ == "__main__":
	import sys
	sys.path.append('/home/tias/Data_science/1_project/gym_training')

import torch
import torch.nn as nn
import torch.nn.functional as F
import copy
import numpy as np

import gym
from utils.replay_buffer import ReplayBuffer, sample_n_unique
from utils.gym_wrapper import PreprocessingWrapper, preprocessing
from torch.utils.tensorboard import SummaryWriter
from utils.general import get_pretrained_model
from datetime import datetime
from collections import deque


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
		self.apply(self._init_weights)

	def forward(self, x):
		return self.network(x)

	def _init_weights(self, module):
		"""
		Source: https://arxiv.org/pdf/1502.01852v1.pdf equation 10 for conv.
		"""
		if isinstance(module, nn.Conv2d):
			_n = module.kernel_size[0]**2 * module.in_channels
			module.weight.data.normal_(mean=0.0, std=(2 / _n)**0.5)
			if module.bias is not None:
				module.bias.data.zero_()
		elif isinstance(module, nn.Linear):
			module.weight.data.normal_(mean=0.0, std=0.15)
			if module.bias is not None:
				module.bias.data.zero_()


class DQN(object):
	def __init__(self, env, config) -> None:

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

		self.pretrained_t = get_pretrained_model(config["model_training"]["load_path"])[1] \
			if config["env"]["use_pretrained_weights"] else 0


	def normalization(self, state):
		return (state / 255).float()


	def play(self, train=False, eval_episode=30):
		"""
		Run game episodes. Used in training (train=True) and evaluation/simulation (train=False) 
		"""
		buffer_size = self.config["hyper_params"]["buffer_size"]
		history = self.config["hyper_params"]["state_history"]
		n_step = self.config["hyper_params"]["nsteps_train"] + self.pretrained_t
		replay_buffer = ReplayBuffer(buffer_size, history)

		t = 0 + self.pretrained_t
		episode = 0
		rewards = deque(maxlen=1000)
		max_q_vals = deque(maxlen=1000)

		while t < n_step:
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
					# Use e-greedy exploration strategy on training
					self.update_epsilon(t)
					action = self.epsilon_greedy(best_action)
				else:
					# On simulation/evaluation use only best action
					action = best_action

				# action = self.env.action_space.sample()

				new_state, reward, done, info = self.env.step(action)

				replay_buffer.store_effect(idx, action, reward, done)
				state = new_state

				episode_reward += reward
				episode_q_vals += np.max(q_vals)  # Max Q over a

				if (train and t > self.config["hyper_params"]["learning_start"] + self.pretrained_t):
					if t % self.config["hyper_params"]["learning_freq"] == 0:
						self.train_step(replay_buffer, t)

					if t % self.config["hyper_params"]["target_update_freq"] == 0:
						# Update target parameter
						self.Q_target.load_state_dict(self.Q.state_dict())

					if t % self.config["model_training"]["eval_freq"] == 0:
						scores = self.play(train=False, eval_episode=self.config["model_training"]["num_episodes_test"])
						# scores : (rewards, max_q_vals)
						self.evaluate(scores, t)

					if t % self.config["model_training"]["saving_freq"] == 0:
						self.save_model(self.config["output"]["model_output"], t)

				if done or t >= n_step:
					episode += 1
					rewards.append(episode_reward)
					max_q_vals.append(episode_q_vals)
					self.tf_add_summary(episode_reward, episode_q_vals, episode)
					break

			# Return 'scores' for evaluation
			if not train and (episode >= eval_episode or t >= n_step):
				del replay_buffer # Delete replay_buffer on evaluation to save memory.
				return (rewards, max_q_vals)
		
		self.save_model(self.config["model_training"]["load_path"], t)


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

		# total_norm = torch.nn.utils.clip_grad_norm_(
		# 	self.Q.parameters(), self.config["model_training"]["clip_val"])

		self.update_lr(t)
		for group in self.optimizer.param_groups:
			group['lr'] = self.lr

		self.optimizer.step()

		# return loss.item(), total_norm.item()


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
		if t <= self.config["hyper_params"]["learning_start"]:
			return val_begin
		elif t > self.config["hyper_params"]["learning_start"] and t <= n_steps + self.config["hyper_params"]["learning_start"]:
			return val_begin + (val_end - val_begin) / n_steps * (t - self.config["hyper_params"]["learning_start"])
		else:
			return val_end


	def save_model(self, path, t):
		FORMAT = '%Y%m%d%H%M%S'
		datenow = datetime.now().strftime(FORMAT)

		# Pathname format: "output_path/model_{datetime}_{t}.weights"
		PATH = path + "/model_{}_{}.weights.pt".format(datenow, t)
		print("		saving ", PATH)
		torch.save(self.Q.state_dict(), PATH)


	def calc_loss(self, q_values: torch.Tensor, target_q_values: torch.Tensor,
				  actions: torch.Tensor, rewards: torch.Tensor, done_mask: torch.Tensor):

		num_actions = self.env.action_space.n
		gamma = self.config["hyper_params"]["gamma"]
		q_ = (F.one_hot(actions.to(int), num_classes=num_actions)
			  * q_values).sum(dim=1)
		q_target = torch.where(
			done_mask, rewards, rewards + gamma * torch.max(target_q_values, axis=1)[0])
		# loss = F.mse_loss(q_, q_target, reduction='mean')
		loss = F.huber_loss(q_, q_target, reduction='mean', delta=1.0)

		return loss


	def evaluate(self, scores, t):
		"""
		Evaluate best action result from trained network.
		"""
		avg_reward = np.mean(scores[0])
		max_reward = np.max(scores[0])
		std_reward = np.std(scores[0])

		avg_q = np.mean(scores[1])
		max_q = np.max(scores[1])
		std_q = np.std(scores[1])

		total_step = self.config["hyper_params"]["nsteps_train"] + self.pretrained_t
		print("Training step {}/{} \t : Score {:.2f} +/-{:.2f} \t date-time: {}".format(t,
			  total_step, avg_reward, std_reward, datetime.now().strftime('%Y-%m-%d|%H:%M:%S')))
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
		self.summary_writer.add_scalar('Reward per episode @training', reward, episode)
		self.summary_writer.add_scalar('Q values per episode @training', Q_val, episode)


	def run_training(self):
		self.play(train=True)


	def run_simulation(self):
		scores = self.play(eval_episode=3)
		print("Score: {}".format(scores[0]))

	
if __name__ == "__main__":
	# For debugging
	import warnings
	import yaml
	from utils.general import join

	warnings.filterwarnings("ignore", module=r"gym")
	yaml.add_constructor("!join", join)

	env = gym.make("ALE/Breakout-v5", render_mode="human")
	# env = gym.make("ALE/Breakout-v5")
	env = PreprocessingWrapper(env, preprocessing)

	config_file = open(
		"/home/tias/Data_science/1_project/gym_training/config/dummy-breakout.yml")
	config = yaml.load(config_file, Loader=yaml.FullLoader)

	model = DQN(env, config)
	path = "/home/tias/Data_science/1_project/gym_training/output/ALE/Breakout-v5/models/model_20220919054801_9750000.weights.pt"
	model.Q.load_state_dict(torch.load(path, map_location="cpu"))

	model.run_simulation()
