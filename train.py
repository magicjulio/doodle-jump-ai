# -*- coding: utf-8 -*-
"""Train a DQN agent for Doodle Jump.

This script is intentionally small and practical:
- uses the existing environment wrapper
- trains a CNN Q-network with experience replay
- saves a checkpoint after the configured duration
"""

from __future__ import annotations

import argparse
import os
import random
import time
from collections import deque
from dataclasses import dataclass
from pathlib import Path

import torch
from torch import nn
from torch.optim import Adam

from environment import DoodleJumpEnv
from model import build_policy


@dataclass
class Transition:
	state: torch.Tensor
	action: int
	reward: float
	next_state: torch.Tensor
	done: bool


class ReplayBuffer:
	def __init__(self, capacity: int):
		self.buffer = deque(maxlen=capacity)

	def append(self, transition: Transition) -> None:
		self.buffer.append(transition)

	def __len__(self) -> int:
		return len(self.buffer)

	def sample(self, batch_size: int) -> list[Transition]:
		return random.sample(self.buffer, batch_size)


def parse_args() -> argparse.Namespace:
	parser = argparse.ArgumentParser(description="Train a DQN agent for Doodle Jump")
	parser.add_argument("--train-steps", type=int, default=50000, help="Number of environment steps to train for")
	parser.add_argument("--save-path", default="checkpoints/doodlejump_dqn.pt", help="Where to save the trained model")
	parser.add_argument("--resume-from", default="", help="Resume training from an existing checkpoint")
	parser.add_argument("--device", default="auto", help="Torch device to use: auto, cpu, cuda, cuda:0, etc.")
	parser.add_argument("--seed", type=int, default=0, help="Random seed; use a negative value to disable seeding")
	parser.add_argument("--batch-size", type=int, default=32)
	parser.add_argument("--replay-size", type=int, default=50000)
	parser.add_argument("--learning-starts", type=int, default=1000)
	parser.add_argument("--train-every", type=int, default=4)
	parser.add_argument("--target-update", type=int, default=1000)
	parser.add_argument("--log-every", type=int, default=1000, help="Print a progress line every N environment steps")
	parser.add_argument("--save-every", type=int, default=50000, help="Save latest checkpoint every N environment steps; 0 disables periodic saves")
	parser.add_argument("--snapshot-every", type=int, default=250000, help="Also save numbered checkpoint snapshots every N environment steps; 0 disables snapshots")
	parser.add_argument("--gamma", type=float, default=0.99)
	parser.add_argument("--lr", type=float, default=1e-4)
	parser.add_argument("--epsilon-start", type=float, default=1.0)
	parser.add_argument("--epsilon-final", type=float, default=0.05)
	parser.add_argument(
		"--epsilon-decay",
		type=int,
		default=0,
		help="Steps over which epsilon decays; use 0 to match --train-steps",
	)
	render_group = parser.add_mutually_exclusive_group()
	render_group.add_argument("--headless", dest="headless", action="store_true", help="Keep rendering off for faster training")
	render_group.add_argument("--visible", dest="headless", action="store_false", help="Render to a visible window during training")
	parser.set_defaults(headless=True)
	return parser.parse_args()


def epsilon_by_step(step: int, start: float, final: float, decay: int) -> float:
	if step >= decay:
		return final
	progress = step / float(decay)
	return start + progress * (final - start)


def to_tensor(observation, device: torch.device) -> torch.Tensor:
	return torch.as_tensor(observation, device=device, dtype=torch.float32).unsqueeze(0)


def optimize_model(policy_net, target_net, optimizer, batch, gamma, device):
	states = torch.stack([transition.state for transition in batch]).to(device)
	actions = torch.tensor([transition.action for transition in batch], device=device).long().unsqueeze(1)
	rewards = torch.tensor([transition.reward for transition in batch], device=device).float().unsqueeze(1)
	next_states = torch.stack([transition.next_state for transition in batch]).to(device)
	dones = torch.tensor([transition.done for transition in batch], device=device).float().unsqueeze(1)

	q_values = policy_net(states).gather(1, actions)
	with torch.no_grad():
		next_actions = policy_net(next_states).argmax(dim=1, keepdim=True)
		next_q_values = target_net(next_states).gather(1, next_actions)
		target_q_values = rewards + gamma * next_q_values * (1.0 - dones)

	loss = nn.functional.smooth_l1_loss(q_values, target_q_values)

	optimizer.zero_grad()
	loss.backward()
	torch.nn.utils.clip_grad_norm_(policy_net.parameters(), 10.0)
	optimizer.step()
	return float(loss.item())


def save_checkpoint(
	path: Path,
	policy_net,
	target_net,
	optimizer,
	env,
	args: argparse.Namespace,
	epsilon_decay: int,
	episode: int,
	start_step: int,
	current_step: int,
	total_reward: float,
	total_loss: float,
	updates: int,
	interrupted: bool = False,
	error_message: str = "",
) -> None:
	path.parent.mkdir(parents=True, exist_ok=True)
	payload = {
		"model_state_dict": policy_net.state_dict(),
		"target_state_dict": target_net.state_dict(),
		"optimizer_state_dict": optimizer.state_dict(),
		"model_class": "DoodleJumpCNN",
		"frame_stack": env.frame_stack,
		"frame_size": env.frame_size,
		"grayscale": env.grayscale,
		"train_steps": args.train_steps,
		"epsilon_start": args.epsilon_start,
		"epsilon_final": args.epsilon_final,
		"epsilon_decay": epsilon_decay,
		"episode": episode,
		"start_step": start_step,
		"final_step": current_step,
		"total_reward": total_reward,
		"total_loss": total_loss,
		"updates": updates,
		"replay_size": args.replay_size,
		"batch_size": args.batch_size,
		"learning_starts": args.learning_starts,
		"train_every": args.train_every,
		"target_update": args.target_update,
		"gamma": args.gamma,
		"lr": args.lr,
		"seed": args.seed,
		"device": str(next(policy_net.parameters()).device),
		"interrupted": interrupted,
		"error_message": error_message,
	}
	tmp_path = path.with_suffix(path.suffix + ".tmp")
	torch.save(payload, tmp_path)
	os.replace(tmp_path, path)


def snapshot_path(save_path: Path, step: int) -> Path:
	return save_path.with_name(f"{save_path.stem}_step_{step:09d}{save_path.suffix}")


def resolve_device(device_name: str) -> torch.device:
	if device_name == "auto":
		return torch.device("cuda" if torch.cuda.is_available() else "cpu")
	return torch.device(device_name)


def main() -> None:
	args = parse_args()
	if args.seed >= 0:
		random.seed(args.seed)
		torch.manual_seed(args.seed)
		if torch.cuda.is_available():
			torch.cuda.manual_seed_all(args.seed)
	device = resolve_device(args.device)
	print(f"device={device}", flush=True)
	env = DoodleJumpEnv(headless=args.headless)
	render_visible = not args.headless
	epsilon_decay = args.train_steps if args.epsilon_decay <= 0 else args.epsilon_decay

	policy_net = build_policy().to(device)
	target_net = build_policy().to(device)
	target_net.load_state_dict(policy_net.state_dict())
	target_net.eval()

	optimizer = Adam(policy_net.parameters(), lr=args.lr)
	replay_buffer = ReplayBuffer(args.replay_size)
	start_step = 0
	total_reward = 0.0
	total_loss = 0.0
	updates = 0
	episode = 1
	start_time = time.perf_counter()

	if args.resume_from:
		checkpoint = torch.load(args.resume_from, map_location=device)
		policy_net.load_state_dict(checkpoint["model_state_dict"])
		target_net.load_state_dict(checkpoint.get("target_state_dict", checkpoint["model_state_dict"]))
		if "optimizer_state_dict" in checkpoint:
			optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
		start_step = int(checkpoint.get("final_step", checkpoint.get("train_steps", 0)))
		if args.epsilon_decay <= 0 and "epsilon_decay" in checkpoint:
			epsilon_decay = int(checkpoint["epsilon_decay"])
		total_reward = float(checkpoint.get("total_reward", 0.0))
		total_loss = float(checkpoint.get("total_loss", 0.0))
		updates = int(checkpoint.get("updates", 0))
		episode = int(checkpoint.get("episode", 1))
		print(f"resumed_from={args.resume_from}")

	observation = env.reset()
	episode_reward = 0.0
	step_count = 0

	output_path = Path(args.save_path)
	last_step = start_step
	interrupted = False
	error_message = ""

	try:
		for step in range(start_step + 1, start_step + args.train_steps + 1):
			last_step = step
			epsilon = epsilon_by_step(step, args.epsilon_start, args.epsilon_final, epsilon_decay)
			state_tensor = torch.as_tensor(observation, dtype=torch.uint8)
			if random.random() < epsilon:
				action = random.randrange(3)
			else:
				policy_net.eval()
				with torch.no_grad():
					action = policy_net.act(state_tensor, greedy=True, device=device)
			policy_net.train()

			next_observation, reward, done, info = env.step(action, render=render_visible)
			next_state_tensor = torch.as_tensor(next_observation, dtype=torch.uint8)
			replay_buffer.append(Transition(state_tensor, action, reward, next_state_tensor, done))

			observation = next_observation
			total_reward += reward
			episode_reward += reward
			step_count += 1

			if len(replay_buffer) >= args.learning_starts and step % args.train_every == 0:
				batch = replay_buffer.sample(args.batch_size)
				loss = optimize_model(policy_net, target_net, optimizer, batch, args.gamma, device)
				total_loss += loss
				updates += 1

			if step % args.target_update == 0:
				target_net.load_state_dict(policy_net.state_dict())

			if done:
				print(
					f"episode={episode} steps={step_count} episode_reward={episode_reward:.2f} "
					f"score={info['score']} epsilon={epsilon:.3f}",
					flush=True,
				)
				observation = env.reset()
				episode_reward = 0.0
				step_count = 0
				episode += 1

			if args.log_every > 0 and (step - start_step) % args.log_every == 0:
				elapsed = time.perf_counter() - start_time
				steps_done = step - start_step
				steps_per_second = steps_done / elapsed if elapsed > 0 else 0.0
				games_played = episode - 1
				print(
					f"progress steps={steps_done}/{args.train_steps} global_step={step} games={games_played} "
					f"updates={updates} epsilon={epsilon:.3f} reward={total_reward:.2f} "
					f"sps={steps_per_second:.2f}",
					flush=True,
				)

			if args.save_every > 0 and (step - start_step) % args.save_every == 0:
				save_checkpoint(
					output_path,
					policy_net,
					target_net,
					optimizer,
					env,
					args,
					epsilon_decay,
					episode,
					start_step,
					step,
					total_reward,
					total_loss,
					updates,
				)
				print(f"saved_checkpoint={output_path} step={step}", flush=True)

			if args.snapshot_every > 0 and (step - start_step) % args.snapshot_every == 0:
				numbered_path = snapshot_path(output_path, step)
				save_checkpoint(
					numbered_path,
					policy_net,
					target_net,
					optimizer,
					env,
					args,
					epsilon_decay,
					episode,
					start_step,
					step,
					total_reward,
					total_loss,
					updates,
				)
				print(f"saved_snapshot={numbered_path} step={step}", flush=True)
	except KeyboardInterrupt:
		interrupted = True
		print("interrupted=true saving_checkpoint=true", flush=True)
	except Exception as error:
		error_message = repr(error)
		print(f"error={error_message} saving_checkpoint=true", flush=True)
		save_checkpoint(
			output_path,
			policy_net,
			target_net,
			optimizer,
			env,
			args,
			epsilon_decay,
			episode,
			start_step,
			last_step,
			total_reward,
			total_loss,
			updates,
			error_message=error_message,
		)
		raise

	save_checkpoint(
		output_path,
		policy_net,
		target_net,
		optimizer,
		env,
		args,
		epsilon_decay,
		episode,
		start_step,
		last_step,
		total_reward,
		total_loss,
		updates,
		interrupted=interrupted,
		error_message=error_message,
	)
	print(f"saved_checkpoint={output_path}")
	print(f"train_steps={args.train_steps}")
	print(f"games_played={episode - 1}")
	print(f"total_reward={total_reward:.2f}")
	print(f"updates={updates}")


if __name__ == "__main__":
	main()
