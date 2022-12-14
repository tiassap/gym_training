import numpy as np
from cv2 import cvtColor, resize, COLOR_RGB2GRAY, INTER_NEAREST
import gym
from gym import spaces
from collections import deque

"""
For walkthrough of the preprocessing, see:
https://danieltakeshi.github.io/2016/11/25/frame-skipping-and-preprocessing-for-deep-q-networks-on-atari-2600-games/
"""


def preprocessing(state):
	"""
	Preprocessing image:
					- Grayscaling
					- Resizing to H:110 x W:84
					- Cropping to H:84 x W:84
	"""
	# Preprocessing
	state = cvtColor(state, COLOR_RGB2GRAY)
	# state = cvtColor(state.astype(np.float32), COLOR_RGB2GRAY)
	state = resize(state, (84, 110), interpolation=INTER_NEAREST)
	state = state[18:102]

	# Adding axis channel and convert to uint8
	state = state[:, :, np.newaxis].astype(np.uint8)
	
	return state

# def preprocessing(state):
# 	"""
# 	Preprocess state (210, 160, 3) image into
# 	a (80, 80, 1) image in grey scale
# 	"""
# 	state = np.reshape(state, [210, 160, 3]).astype(np.float32)

# 	# grey scale
# 	state = state[:, :, 0] * 0.299 + state[:, :, 1] * 0.587 + state[:, :, 2] * 0.114

# 	# karpathy
# 	state = state[31:191]  # crop
# 	state = state[::2,::2] # downsample by factor of 2

# 	state = state[:, :, np.newaxis]

# 	return state.astype(np.uint8)
	

class PreprocessingWrapper(gym.Wrapper):
	def __init__(self, env, prepro):
		super().__init__(env)
		self._obs_buffer = deque(maxlen=2)
		self._skip = 4
		self.prepro = prepro
		self.observation_space = spaces.Box(low=0, high=255, shape=(84, 84, 1), dtype=np.uint8)
		self.overwrite_render = True
		self.viewer = None

	def step(self, action):
		total_reward = 0.0
		done = None

		# Skipping 4 frames
		for _ in range(self._skip):
			obs, reward, done, info = self.env.step(action)
			self._obs_buffer.append(obs)
			total_reward += reward
			if done:
				break
		
		# Taking pixel by pixel maximum over two frames.
		obs = np.max(np.stack(self._obs_buffer), axis=0) 

		# Preprocessing image: grayscaling, resizing, cropping
		obs = self.prepro(obs)

		# import pdb;pdb.set_trace()
		return obs, total_reward, done, info


	def reset(self):
		self._obs_buffer.clear()
		self.obs = self.prepro(self.env.reset())
		self._obs_buffer.append(self.obs)

		# import pdb;pdb.set_trace()
		return self.obs


if __name__ == "__main__":
	import numpy as np

	image = np.random.randint(0, 255, (210, 160, 3)).astype(np.uint8)
	image = preprocessing(image)
	print(image.shape)
	print(image.dtype)