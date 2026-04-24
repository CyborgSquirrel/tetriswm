import json
from PIL import Image
import os.path
import os

import gymnasium as gym
import numpy as np
from tetris_gymnasium.components.tetromino import Tetromino
from tetris_gymnasium.envs import Tetris

DEST_PATH_FMT = "data/traj1/{traj}-{step}-{name}.{ext}"


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

  for traj_idx in range(100):
    env.reset(seed=rng.integers(0,2**32-1).item())
    traj_len = rng.integers(10,15).item()
    for step_idx in range(traj_len):
      # NOTE(andrei): Seems like the active Tetromino mask is not as precise as
      # I would like. Maybe it would be better to write our own Tetris
      # implementation after all?

      meta = {}

      if step_idx > 0:
        action = env.action_space.sample()
        _im, reward, game_over, truncated, _info = env.step(action)
        meta["action"] = int(action)
        meta["reward"] = int(reward)
        meta["game_over"] = game_over
        meta["truncated"] = truncated

      # im - H,W,C
      im = env.render()

      im_path = DEST_PATH_FMT.format(
        traj=traj_idx,
        step=step_idx,
        name="im",
        ext="png",
      )
      os.makedirs(os.path.dirname(im_path), exist_ok=True)
      Image.fromarray(im).save(im_path)

      meta_path = DEST_PATH_FMT.format(
        traj=traj_idx,
        step=step_idx,
        name="meta",
        ext="json",
      )
      os.makedirs(os.path.dirname(meta_path), exist_ok=True)
      with open(meta_path, "w") as f:
        json.dump(meta, f)

if __name__ == "__main__":
  main()
