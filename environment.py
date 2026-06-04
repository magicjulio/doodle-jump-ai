# -*- coding: utf-8 -*-
"""Lightweight RL wrapper around the Doodle Jump game.

This exposes a small Gym-like interface without depending on keyboard events.
"""

from collections import deque

import numpy as np

from main import Game
import settings as config


class DoodleJumpEnv:
	"""Discrete-action environment for an agent.

	Actions:
	0 = left
	1 = no-op
	2 = right
	"""

	def __init__(self, frame_size=(84, 84), frame_stack=4, grayscale=True, headless=True):
		self.game = Game(headless=headless)
		self.frame_size = frame_size
		self.frame_stack = frame_stack
		self.grayscale = grayscale
		self.headless = headless
		self._frames = deque(maxlen=frame_stack)
		self._last_score = 0
		self._last_camera_y = 0
		self._best_landing_y = None

	def set_headless(self, headless:bool) -> None:
		"""Toggle visible rendering without changing the rest of the API."""
		self.headless = headless
		self.game.set_headless(headless)

	def _capture_frame(self):
		return self.game.get_observation(
			size=self.frame_size,
			grayscale=self.grayscale,
		)

	def _stack_frames(self, frame):
		self._frames.append(frame)
		while len(self._frames) < self.frame_stack:
			self._frames.append(frame)
		if self.frame_stack == 1:
			return frame
		return np.stack(list(self._frames), axis=0)

	def reset(self):
		"""Reset the game and return the first stacked observation."""
		self.game.reset()
		self.game._render_loop(present=not self.headless)
		self._frames.clear()
		frame = self._capture_frame()
		self._frames.extend([frame] * self.frame_stack)
		self._last_score = self.game.score
		self._last_camera_y = self.game.camera.state.y
		self._best_landing_y = None
		return self._stack_frames(frame)

	def _compute_reward(self, previous_score: int, previous_camera_y: int, action: int, done: bool) -> tuple[float, dict]:
		"""Shape reward around upward progress and successful landings."""
		current_camera_y = self.game.camera.state.y
		current_score = self.game.score
		landed = self.game.player.landed_this_frame
		bonus_landed = self.game.player.bonus_this_frame
		new_high_landing = False

		height_delta = previous_camera_y - current_camera_y
		score_gain = max(0, current_score - previous_score)

		reward = (height_delta * config.REWARD_HEIGHT_SCALE) + (score_gain * config.REWARD_SCORE_BONUS) + config.REWARD_STEP_PENALTY
		if landed:
			landing_y = self.game.player.rect.y
			if self._best_landing_y is None or landing_y < self._best_landing_y:
				new_high_landing = True
				self._best_landing_y = landing_y
		if new_high_landing:
			reward += config.REWARD_LANDING_BONUS
		if bonus_landed and new_high_landing:
			reward += config.REWARD_BONUS_LANDING_BONUS
		if action == 1:
			reward += config.REWARD_NOOP_PENALTY
		if done:
			reward += config.REWARD_DEATH_PENALTY

		self._last_camera_y = current_camera_y

		return reward, {
				"height_delta": height_delta,
				"score_gain": score_gain,
				"landed": landed,
				"new_high_landing": new_high_landing,
				"bonus_landed": bonus_landed,
				"last_camera_y": self._last_camera_y,
			}

	def step(self, action, render=None):
		"""Advance the environment one frame.

		Returns: observation, reward, done, info
		"""
		previous_score = self.game.score
		previous_camera_y = self._last_camera_y
		if render is None:
			render = not self.headless
		self.game.step(action=action, render=render)
		frame = self._capture_frame()
		observation = self._stack_frames(frame)

		done = self.game.player.dead
		reward, reward_info = self._compute_reward(previous_score, previous_camera_y, action, done)
		self._last_score = self.game.score

		info = {
			"score": self.game.score,
			"dead": done,
			"action": action,
			**reward_info,
		}
		return observation, reward, done, info
