# -*- coding: utf-8 -*-
"""PyTorch Q-network for Doodle Jump.

This is a compact CNN meant for stacked grayscale frames.
Input shape: (batch, 4, 84, 84)
Output shape: (batch, 3)
"""

from __future__ import annotations

import torch
from torch import nn


class DoodleJumpCNN(nn.Module):
	"""Small CNN backbone that outputs action values for DQN.

	The architecture is intentionally modest so it runs well on CPU while still
	having enough capacity to learn the simple spatial patterns in Doodle Jump.
	"""

	def __init__(self, in_channels: int = 4, num_actions: int = 3):
		super().__init__()
		self.in_channels = in_channels
		self.num_actions = num_actions

		self.encoder = nn.Sequential(
			nn.Conv2d(in_channels, 16, kernel_size=8, stride=4),
			nn.ReLU(inplace=True),
			nn.Conv2d(16, 32, kernel_size=4, stride=2),
			nn.ReLU(inplace=True),
			nn.Conv2d(32, 64, kernel_size=3, stride=1),
			nn.ReLU(inplace=True),
			nn.Flatten(),
		)

		self.head = nn.Sequential(
			nn.Linear(64 * 7 * 7, 256),
			nn.ReLU(inplace=True),
			nn.Linear(256, num_actions),
		)

	def forward(self, x: torch.Tensor) -> torch.Tensor:
		"""Return action values for a batch of stacked frames."""
		if x.ndim == 3:
			x = x.unsqueeze(0)
		if x.dtype == torch.uint8:
			x = x.float().div(255.0)
		else:
			x = x.float()
			if x.numel() and x.detach().max() > 1.0:
				x = x.div(255.0)
		features = self.encoder(x)
		return self.head(features)

	@torch.no_grad()
	def act(self, observation, greedy: bool = True, device: str | torch.device | None = None) -> int:
		"""Convert one observation into an action index.

		If greedy is True, returns argmax action values. Otherwise samples from the
		softmax distribution, which is useful later for exploration.
		"""
		if not torch.is_tensor(observation):
			observation = torch.as_tensor(observation)
		if observation.ndim == 3:
			observation = observation.unsqueeze(0)
		if device is not None:
			observation = observation.to(device)
		logits = self.forward(observation)
		if greedy:
			return int(torch.argmax(logits, dim=1).item())
		probabilities = torch.softmax(logits, dim=1)
		return int(torch.multinomial(probabilities, num_samples=1).item())


def build_policy(in_channels: int = 4, num_actions: int = 3) -> DoodleJumpCNN:
	"""Factory for the default Doodle Jump Q-network."""
	return DoodleJumpCNN(in_channels=in_channels, num_actions=num_actions)
