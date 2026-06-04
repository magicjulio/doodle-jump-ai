# -*- coding: utf-8 -*-
"""Run the current policy against the live game.

This is a lightweight sanity check for the full observation -> model -> action
pipeline. It is also useful later as an evaluation harness.
"""

from __future__ import annotations

import argparse
import time

import torch

from environment import DoodleJumpEnv
from model import build_policy


def parse_args() -> argparse.Namespace:
	parser = argparse.ArgumentParser(description="Run a Doodle Jump policy rollout")
	parser.add_argument("--steps", type=int, default=300, help="Number of environment steps to run")
	parser.add_argument("--checkpoint", default="", help="Optional model checkpoint to evaluate")
	parser.add_argument("--visible", action="store_true", help="Render to a visible window instead of headless mode")
	parser.add_argument("--sample", action="store_true", help="Sample from softmax action values instead of using argmax")
	parser.add_argument("--device", default="auto", help="Torch device for inference: auto, cpu, cuda, cuda:0, etc.")
	return parser.parse_args()


def resolve_device(device_name: str) -> torch.device:
	if device_name == "auto":
		return torch.device("cuda" if torch.cuda.is_available() else "cpu")
	return torch.device(device_name)


def main() -> None:
	args = parse_args()
	device = resolve_device(args.device)
	env = DoodleJumpEnv(headless=not args.visible)
	policy = build_policy().to(device)
	if args.checkpoint:
		checkpoint = torch.load(args.checkpoint, map_location=device)
		policy.load_state_dict(checkpoint["model_state_dict"])
		print(f"loaded_checkpoint={args.checkpoint}")
	policy.eval()

	observation = env.reset()
	start = time.perf_counter()
	total_reward = 0.0
	steps = 0

	for _ in range(args.steps):
		action = policy.act(observation, greedy=not args.sample, device=device)
		observation, reward, done, info = env.step(action, render=args.visible)
		total_reward += reward
		steps += 1
		if done:
			observation = env.reset()

	elapsed = time.perf_counter() - start
	steps_per_second = steps / elapsed if elapsed > 0 else 0.0

	print(f"steps={steps}")
	print(f"elapsed_sec={elapsed:.3f}")
	print(f"steps_per_sec={steps_per_second:.2f}")
	print(f"total_reward={total_reward:.2f}")
	print(f"last_score={info['score']}")


if __name__ == "__main__":
	main()
