import argparse
from einops import rearrange
import itertools as itt
import os.path
import os

import numpy as np
from tetris import tetris
import h5py
from tqdm import tqdm

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

  keys = {}
  keys["pixels"] = { "image": True }
  keys["tetris_grid_mask"] = { "image": True }
  keys["tetris_mino_mask"] = { "image": True }
  keys["action"] = {}
  keys["ep_len"] = {}
  keys["ep_idx"] = {}
  keys["ep_offset"] = {}

  data = { key: [] for key in keys.keys() }

  actions = np.array(tetris.Action)

  ep_idx_it = 0
  ep_offset_it = 0

  for traj_idx in tqdm(range(traj_num)):
    env = tetris.Tetris()

    ep_idx = ep_idx_it
    ep_idx_it += 1

    ep_offset = ep_offset_it
    for step_idx in range(traj_len):
      data["ep_idx"].append(ep_idx)

      ep_offset_it += 1

      pixels = env.render()
      data["pixels"].append(rearrange(pixels, "w h c -> h w c"))

      tetris_grid_mask = (env.grid.copy() != 0)
      tetris_mino_mask = np.zeros_like(tetris_grid_mask)
      if env.current is not None:
        for x,y in env.current.coords:
          tetris_mino_mask[x,y] = 1
      tetris_grid_mask = tetris_grid_mask | tetris_mino_mask
      data["tetris_grid_mask"].append(tetris_grid_mask.astype(np.uint8))
      data["tetris_mino_mask"].append(tetris_mino_mask.astype(np.uint8))

      # TODO: Le-WM actions seem to range between [-1;1] in the training
      # datasets, maybe do the same here.

      # NOTE: Action taken in the current step, leading to the next step.
      action_onehot = np.zeros(len(tetris.Action))
      if step_idx < traj_len-1:
        action = rng.choice(actions)
        action_onehot[action-1] = 1
        env.step(action)

      data["action"].append(action_onehot)

      if env.state == tetris.GameState.OVER:
        break
    ep_len = ep_offset_it - ep_offset

    data["ep_len"].append(ep_len)
    data["ep_offset"].append(ep_offset)

  # Combine everything
  data = {
    key: np.stack(value, axis=0)
    for key, value in data.items()
  }

  # Write to HDF5 file
  with h5py.File(out_path, "w") as f:
    for key, value in data.items():
      key_spec = keys[key]

      dataset_args = {}

      if key_spec.get("image", False):
        dataset_args["chunks"] = (1,) + value.shape[1:]
        dataset_args["compression"] = "gzip"

      f.create_dataset(key, data=value, **dataset_args)

if __name__ == "__main__":
  main()
