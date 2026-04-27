import json
from PIL import Image
import os.path
import os

import gymnasium as gym
import numpy as np
from tetris_gymnasium.components.tetromino import Tetromino
from tetris_gymnasium.envs import Tetris
import h5py

# TODO: better game layout that is more economical resolution-wise

def main():
  tetrominoes = [
    # I
    Tetromino(
      0,
      [0, 240, 240],
      np.array(
        [[0, 0, 0, 0], [1, 1, 1, 1], [0, 0, 0, 0], [0, 0, 0, 0]], dtype=np.uint8
      ),
    ),
  ]

  # Create an instance of Tetris
  env = gym.make(
    "tetris_gymnasium/Tetris",
    render_mode="rgb_array",
    tetrominoes=tetrominoes,
  )
  env: Tetris

  rng: np.random.Generator = np.random.default_rng(seed=42)

  # print(env.unwrapped.actions)

  out_path = "data/traj_overfit.h5"
  traj_num = 1
  traj_len = 15

  data = {}
  data["pixels"] = []
  data["action"] = []

  # meta = {}
  for traj_idx in range(traj_num):
    env.reset(seed=rng.integers(0,2**32-1).item())
    for step_idx in range(traj_len):
      # NOTE(andrei): Seems like the active Tetromino mask is not as precise as
      # I would like. Maybe it would be better to write our own Tetris
      # implementation after all?

      # im - H,W,C
      pixels = env.render()
      data["pixels"].append(pixels)

      # TODO: Le-WM actions seem to range between [-1;1] in the training
      # datasets, maybe do the same here.

      # NOTE: Action taken in the current step, leading to the next step.
      if step_idx >= traj_len-1:
        action = 0
      else:
        true_action = env.action_space.sample()
        action = true_action + 1  # NOTE: Saving 0 as trajectory splitter

        # TODO: Maybe use this stuff for something.
        _im, _reward, _game_over, _truncated, _info = env.step(true_action)

      data["action"].append(action)

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
