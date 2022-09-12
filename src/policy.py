import torch
import torch.nn as nn
import torch.distributions as ptd

from abc import ABC, abstractmethod
from utils.network_utils import device, np2torch


class CategoricalPolicy(nn.Module):

	def __init__(self, network):
		nn.Module.__init__(self)
		self.network = network

	def act(self, observations):
		observations = np2torch(observations)
		dist = self.action_distribution(observations)
		sampled_actions = dist.sample().detach().cpu().numpy()
		return sampled_actions

	def action_distribution(self, observations):
		"""
		Args:
			observations (torch.Tensor):  observation of states from the environment
										(shape [batch size, dim(observation space)])

		Returns:
			distribution (torch.distributions.Categorical): represent the conditional distribution over
															actions given a particular observation

		"""
		return ptd.Categorical(logits=self.network(observations))


class GaussianPolicy(nn.Module):

    def __init__(self, network, action_dim):
        nn.Module.__init__(self)
        self.network = network
        self.log_std = nn.Parameter(torch.zeros(action_dim).to(device)).to(device)

    def act(self, observations):
        observations = np2torch(observations)
        dist = self.action_distribution(observations)
        sampled_actions = dist.sample().detach().cpu().numpy()
        return sampled_actions

    def std(self):
        """
        Returns:
            std (torch.Tensor):  the standard deviation for each dimension of the policy's actions
                                (shape [dim(action space)])
        """
        return torch.exp(self.log_std)

    def action_distribution(self, observations):
        """
        Args:
            observations (torch.Tensor):  observation of states from the environment
                                        (shape [batch size, dim(observation space)])

        Returns:
            distribution (torch.distributions.Distribution): a pytorch distribution representing
                a diagonal Gaussian distribution whose mean (loc) is computed by
                self.network and standard deviation (scale) is self.std()
        """
        cov = torch.diag(self.std()**2)
        return ptd.MultivariateNormal(loc=self.network(observations), covariance_matrix=cov)
