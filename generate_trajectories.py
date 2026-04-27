"""
Trajectory data collection for LeWorldModel training.

Data format (per trajectory, stored in dest_dir/):
  {traj}-{step}-im.png    — frame observed BEFORE taking the action at this step
  {traj}-{step}-meta.json — {"action": int, "reward": float, "game_over": bool}
  {traj}-{T}-im.png       — terminal frame (game over / truncated), no meta file

WHY this convention (action = action taken FROM this state):
  LeWM predicts emb[t+1] from (emb[t], action[t]).
  action[t] must be the action that transitions FROM state t TO state t+1.
  The previous convention stored "action that led TO state t", which is off by one.
"""

import argparse
import json
import os
from pathlib import Path

import numpy as np
from PIL import Image
from tetris_gymnasium.envs import Tetris
from tqdm import tqdm


def collect_one_trajectory(env, rng, max_steps: int):
    """
    Run one episode and return (frames, actions, rewards, dones).

    frames[t]  = observation BEFORE action[t]
    frames[-1] = terminal observation (no action taken from it)
    """
    frames, actions, rewards, dones = [], [], [], []

    env.reset(seed=int(rng.integers(0, 2**32 - 1)))

    for _ in range(max_steps):
        frame = env.render()          # observe BEFORE acting
        action = env.action_space.sample()
        _, reward, game_over, truncated, _ = env.step(action)

        frames.append(frame)
        actions.append(int(action))
        rewards.append(float(reward))
        dones.append(bool(game_over or truncated))

        if game_over or truncated:
            break

    # Append the terminal observation (no action taken from here)
    frames.append(env.render())

    return frames, actions, rewards, dones


def save_trajectory(frames, actions, rewards, dones, traj_idx: int, dest_dir: Path):
    dest_dir.mkdir(parents=True, exist_ok=True)

    for step, (frame, action, reward, done) in enumerate(
        zip(frames[:-1], actions, rewards, dones)
    ):
        # Save frame
        Image.fromarray(frame).save(dest_dir / f"{traj_idx}-{step}-im.png")
        # Save meta: action taken FROM this state
        with open(dest_dir / f"{traj_idx}-{step}-meta.json", "w") as f:
            json.dump({"action": action, "reward": reward, "game_over": done}, f)

    # Save terminal frame (no meta — no action taken from terminal state)
    terminal_step = len(actions)
    Image.fromarray(frames[-1]).save(dest_dir / f"{traj_idx}-{terminal_step}-im.png")


def already_done(traj_idx: int, dest_dir: Path) -> bool:
    """Skip trajectories that were already generated (allows resuming)."""
    return any(dest_dir.glob(f"{traj_idx}-*-im.png"))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dest",       default="data/train",
                        help="Output directory")
    parser.add_argument("--n_traj",     type=int, default=5000,
                        help="Number of trajectories to generate")
    parser.add_argument("--max_steps",  type=int, default=200,
                        help="Max steps per trajectory (episode ends early on game over)")
    parser.add_argument("--seed",       type=int, default=42)
    args = parser.parse_args()

    dest_dir = Path(args.dest)
    rng = np.random.default_rng(seed=args.seed)

    # Use all 7 standard tetrominoes (default env)
    env = Tetris(render_mode="rgb_array")
    n_actions = env.action_space.n
    env.reset()
    print(f"Action space: Discrete({n_actions})")
    print(f"Frame shape:  {env.render().shape}")
    print(f"Generating {args.n_traj} trajectories → {dest_dir}/")

    skipped = 0
    for traj_idx in tqdm(range(args.n_traj), desc="trajectories"):
        if already_done(traj_idx, dest_dir):
            skipped += 1
            continue

        frames, actions, rewards, dones = collect_one_trajectory(
            env, rng, args.max_steps
        )
        save_trajectory(frames, actions, rewards, dones, traj_idx, dest_dir)

    env.close()

    n_files = len(list(dest_dir.glob("*-im.png")))
    print(f"\nDone. {n_files} frames saved to {dest_dir}/  (skipped {skipped} existing)")
    print(f"Update train.py --n_actions to {n_actions} if needed.")


if __name__ == "__main__":
    main()
