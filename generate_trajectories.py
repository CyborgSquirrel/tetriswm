import logging
import contextlib as ctl
from einops import rearrange
import os
import os.path
from pathlib import Path

import numpy as np
from tetris import tetris
import h5py
from tqdm import tqdm
import hydra
from omegaconf import DictConfig
from utils import LazyDataset

log = logging.getLogger(__name__)


@hydra.main(config_path="conf/tetris/", config_name="overfit", version_base=None)
def main(cfg: DictConfig):
  rng: np.random.Generator = np.random.default_rng(seed=cfg.seed)

  actions = set(tetris.Action)
  for name in cfg.remove_actions:
    actions.discard(tetris.Action[name])
  actions = np.array(list(actions))

  keys = {}
  keys["pixels"] = { "image": True }
  keys["tetris_grid_mask"] = { "image": True }
  keys["tetris_mino_mask"] = { "image": True }
  keys["tetris_queue"] = {}
  keys["action"] = {}
  keys["ep_len"] = {}
  keys["ep_idx"] = {}
  keys["ep_offset"] = {}
  keys["step_idx"] = {}
  keys["state_placed_grid"] = {}
  keys["state_mino_coords"] = {}
  keys["state_mino_half"] = {}
  keys["state_mino_origin"] = {}
  keys["state_mino_rotations"] = {}
  keys["state_mino_tile_idx"] = {}

  # Write to HDF5 file
  with ctl.ExitStack() as stack:
    path = Path.home() / ".stable_worldmodel" / f"{cfg.out}.h5"
    f = stack.enter_context(h5py.File(path, "w"))

    # Create (lazy) datasets
    data = {}
    for key, key_spec in keys.items():
      def init_hook(
        ds,
        *,
        shape,
        dtype,
        # hack for ridiculous Python binding rules
        key_spec=key_spec,
      ):
        if key_spec.get("image", False):
          ds.dataset_kwargs["chunks"] = (1, *shape)
          ds.dataset_kwargs["compression"] = "gzip"

      data[key] = stack.enter_context(LazyDataset(f, key, init_hook=init_hook))

    # Generate data
    ep_idx_it = 0
    ep_offset_it = 0

    for traj_idx in tqdm(range(cfg.traj_num)):
      env = tetris.Tetris.from_config(
        cfg.tetris,
        rng=np.random.default_rng(tetris.rng_derive_seed(rng)),
      )

      ep_idx = ep_idx_it
      ep_idx_it += 1

      ep_offset = ep_offset_it
      for step_idx in range(cfg.traj_len):
        data["ep_idx"].append(ep_idx)
        data["step_idx"].append(step_idx)

        ep_offset_it += 1

        pixels = env.render()
        data["pixels"].append(rearrange(pixels, "w h c -> h w c"))

        data["tetris_grid_mask"].append((env.grid != 0).astype(np.uint8))
        data["tetris_mino_mask"].append(env.mino_mask.astype(np.uint8))
        data["tetris_queue"].append(env.queue.astype(np.uint8))

        dyn = env.dynamics
        data["state_placed_grid"].append(dyn.grid.copy())
        data["state_mino_coords"].append(dyn.current.coords.astype(np.int16))
        data["state_mino_half"].append(dyn.current.half.astype(np.int16))
        data["state_mino_origin"].append(dyn.current.origin.astype(np.int16))
        data["state_mino_rotations"].append(np.int16(dyn.current.rotations))
        data["state_mino_tile_idx"].append(np.int16(dyn.tetris.tile_set.obj_to_oid[dyn.current.tile]))

        # TODO: Le-WM actions seem to range between [-1;1] in the training
        # datasets, maybe do the same here.

        # NOTE: Action taken in the current step, leading to the next step.
        action_onehot = np.zeros(len(tetris.Action))
        if step_idx < cfg.traj_len-1:
          action = rng.choice(actions)
          action_onehot[action-1] = 1
          env.step(action)

        data["action"].append(action_onehot)

        if env.dynamics.state == tetris.GameState.OVER:
          break
      ep_len = ep_offset_it - ep_offset

      data["ep_len"].append(ep_len)
      data["ep_offset"].append(ep_offset)


if __name__ == "__main__":
  main()
