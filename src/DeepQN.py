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

class NN_VFA(nn.Module):
	def __init__(self, env, config) -> None:
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
		# Initialize model Q and Q_target
		self.Q = NN_VFA(env, config)
		self.Q_target = copy.deepcopy(self.Q)
		self.device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

		self.Q = self.Q.to(self.device)
		self.Q_target = self.Q_target.to(self.device)
		self.optimizer = torch.optim.Adam(self.Q.parameters())
		self.lr = config["hyper_params"]["lr_begin"]

		self.env = env
		self.config = config

		self.summary_writer = SummaryWriter(self.config["output"]["output_path"], max_queue=1e5)

	def play(self, train=False, eval=False):
		buffer_size = self.config["hyper_params"]["buffer_size"]
		history = self.config["hyper_params"]["state_history"]
		n_step = self.config["hyper_params"]["nsteps_train"]
		replay_buffer = ReplayBuffer(buffer_size, history)

		eval_episode = 50
		t = 0
		episode = 0
		rewards = []

		while t <= n_step:
			state = self.env.reset()
			episode_reward = 0

			while True:
				t += 1
				
				idx =replay_buffer.store_frame(state)

				# Concatenate 4 frames of recent observation. Add padding 0 if necessary.
				q_input = replay_buffer.encode_recent_observation()

				best_action, q_vals = self.get_best_action(q_input)

				if train:
					# Use e-greedy exploration strategy
					action = self.epsilon_greedy(best_action)
				else:
					action = best_action

				new_state, reward, done, info = self.env.step(action)

				replay_buffer.store_effect(idx, action, reward, done)
				state = new_state

				if (train and t > self.config["hyper_params"]["learning_start"]):
					if t % self.config["hyper_params"]["learning_freq"] == 0:
						self.train_step(replay_buffer, t)

					if t % self.config["hyper_params"]["target_update_freq"] == 0:
						# Update target parameter
						self.Q_target.load_state_dict(self.Q.state_dict())

					if t % self.config["model_training"]["saving_freq"] == 0:
						self.save_model()

					if t % self.config["hyper_params"]["eval_freq"] == 0:
						scores = self.play(eval=True)
						self.evaluate(scores, t)

					if t % self.config["model_training"]["saving_freq"] == 0:
						self.save_model(t)
				
				episode_reward += reward

				if done or t >= n_step:
					episode += 1
					rewards += [episode_reward]
					break
			
			# Return 'scores' for evaluation
			if eval and episode >= eval_episode:
				return rewards 


	# def epsilon_greedy(self, state):
	def epsilon_greedy(self, action):
		epsilon = self.config["model_training"]["soft_epsilon"]
		if np.random.random() < epsilon:
			return self.env.action_space.sample()
		# else:
		# 	return self.get_best_action(state)[0]
		else:
			return  action

	def get_best_action(self, state: torch.Tensor) -> Tuple[int, np.ndarray]:
		"""
		Return best action

		Args:
			state: 4 consecutive observations from gym
		Returns:
			action: (int)
			action_values: (np array) q values for all actions
		"""
		with torch.no_grad():
			s = torch.tensor(state, dtype=torch.uint8, device=self.device).unsqueeze(0)
			# s = self.process_state(s)
			action_values = self.get_q_values(s, 'q_network').squeeze().to('cpu').tolist()
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
		states, actions, rewards, next_states, dones = replay_buffer.sample(batch_size = 32)

		self.optimizer.zero_grad()

		q_values = self.get_q_values(states, 'q_network')

		with torch.no_grad():
			target_q_values = self.get_q_values(next_states, 'target_network')

		loss = self.calc_loss(q_values, target_q_values, actions, rewards, dones)

		loss.backward()

		self.update_lr(t)
		for group in self.optimizer.param_groups:
			group['lr'] = self.lr
		
		self.optimizer.step()

	def update_lr(self, t):
		lr_begin = self.config["hyper_params"]["lr_begin"]
		lr_end = self.config["hyper_params"]["lr_end"]
		nsteps = self.config["hyper_params"]["lr_nsteps"]

		if t <= nsteps:
			self.lr = lr_begin + (lr_end - lr_begin) / nsteps * t
		else:
			self.lr = lr_end


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
		gamma =  self.config["hyper_params"]["gamma"]
		q_ = (F.one_hot(actions.to(int), num_classes = num_actions) * q_values).sum(dim=1)
		q_target = torch.where(done_mask, rewards, rewards + gamma * torch.max(target_q_values, axis=1)[0])
		loss = F.mse_loss(q_, q_target, reduction='mean')
		
		return loss

	def evaluate(self, scores, t):
		avg_score = np.mean(scores)
		std_score = np.std(scores)
		total_step = self.config["hyper_params"]["nsteps_train"]
		print("Training step {}/{} \t : Score {} +/- {}".format(t, total_step, avg_score, std_score))

	def add_summary(self):
		pass

	def run_training(self):
		self.play(train=True)

	def run_simulation(self):
		self.play()

if __name__ == "__main__":
	# For debugging

	import warnings
	import yaml
	from utils.general import join

	warnings.filterwarnings("ignore", module=r"gym")
	yaml.add_constructor("!join", join)

	env = gym.make("Breakout-v4")
	env = PreprocessingWrapper(env, preprocessing)

	config_file = open("/home/tias/Data_science/1_project/gym_training/config/breakout.yml")
	config = yaml.load(config_file, Loader=yaml.FullLoader)

	model = DQN(env, config)
	import pdb; pdb.set_trace()
