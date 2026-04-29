import argparse
import os.path
import os

import numpy as np
from tetris import tetris
import h5py

def parse_args():
  p = argparse.ArgumentParser()
  p.add_argument("out")
  p.add_argument("--traj-num", type=int, default=1)
  p.add_argument("--traj-len", type=int, default=50)
  p.add_argument("--seed", type=int, default=42)
  return p.parse_args()

def main():
  args = parse_args()
  rng: np.random.Generator = np.random.default_rng(seed=args.seed)

  out_path = args.out
  traj_num = args.traj_num
  traj_len = args.traj_len

  data = {}
  data["pixels"] = []
  data["action"] = []

  actions = np.array(tetris.Action)

  # meta = {}
  for traj_idx in range(traj_num):
    env = tetris.Tetris()
    for step_idx in range(traj_len):
      # pixels - W,H,C
      pixels = env.render()
      data["pixels"].append(pixels.transpose(1,0,2))

      # TODO: Le-WM actions seem to range between [-1;1] in the training
      # datasets, maybe do the same here.

      # NOTE: Action taken in the current step, leading to the next step.
      if step_idx >= traj_len-1:
        action = 0
      else:
        true_action = rng.choice(actions)
        action = true_action + 1  # NOTE: Saving 0 as trajectory splitter

        env.step(true_action)

      data["action"].append(action)

      if env.state == tetris.GameState.OVER:
        break

  # Combine everything
  data = {
    key: np.stack(value, axis=0)
    for key, value in data.items()
  }

  # Write to HDF5 file
  with h5py.File(out_path, "w") as f:
    for key, value in data.items():
      f[key] = value

if __name__ == "__main__":
  main()
