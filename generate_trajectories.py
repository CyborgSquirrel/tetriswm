import logging
import contextlib as ctl
from einops import rearrange
import os

import numpy as np
from tetris import tetris
import h5py
from tqdm import tqdm
import hydra
from hydra.utils import instantiate
from omegaconf import DictConfig

log = logging.getLogger(__name__)


class LazyDataset:
  def __init__(self, f, name, *, dataset_kwargs=None, init_hook=None):
    self.f = f
    self.name = name
    if dataset_kwargs is None:
      dataset_kwargs = {}
    self.dataset_kwargs = dataset_kwargs
    self.init_hook = init_hook
    self.entered = False
    self.initted = False

  def __enter__(self):
    if self.entered:
      raise RuntimeError(f"Tried to enter {repr(type(self))} twice")
    self.entered = True
    return self

  def __exit__(self, exc_type, exc_value, traceback):
    self.entered = False
    self.dataset.resize(self.idx, axis=0)

  def _ensure_initted(self, *, shape=None, dtype=None):
    if self.initted:
      return
    if shape is None or dtype is None:
      raise ValueError()

    if self.init_hook is not None:
      self.init_hook(self, shape=shape, dtype=dtype)

    if (
      self.dataset_kwargs.get("maxshape") is not None
      or self.dataset_kwargs.get("dtype") is not None
      or self.dataset_kwargs.get("shape") is not None
    ):
      raise ValueError()

    self.dataset = self.f.create_dataset(
      self.name,
      data=np.empty((1, *shape)),
      **self.dataset_kwargs,
      maxshape=(None, *shape),
      dtype=dtype,
    )
    self.idx = 0
    self.initted = True

  def append(self, item):
    if isinstance(item, np.ndarray):
      shape = item.shape
      dtype = item.dtype
    else:
      shape = ()
      dtype = type(item)
    self._ensure_initted(shape=shape, dtype=dtype)

    if self.idx >= len(self.dataset):
      self.dataset.resize(len(self.dataset)*2, axis=0)

    self.dataset[self.idx] = item
    self.idx += 1


@hydra.main(config_path="config/tetris/", config_name="overfit", version_base=None)
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
  keys["action"] = {}
  keys["ep_len"] = {}
  keys["ep_idx"] = {}
  keys["ep_offset"] = {}

  # Write to HDF5 file
  with ctl.ExitStack() as stack:
    f = stack.enter_context(h5py.File(cfg.out, "w"))

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
      mino_generator = instantiate(
        cfg.tetris.mino_generator,
        rng=np.random.default_rng(tetris.rng_derive_seed(rng)),
      )
      env = tetris.Tetris(
        width=cfg.tetris.width,
        height=cfg.tetris.height,
        mino_generator=mino_generator,
      )

      ep_idx = ep_idx_it
      ep_idx_it += 1

      ep_offset = ep_offset_it
      for step_idx in range(cfg.traj_len):
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
        if step_idx < cfg.traj_len-1:
          action = rng.choice(actions)
          action_onehot[action-1] = 1
          env.step(action)

        data["action"].append(action_onehot)

        if env.state == tetris.GameState.OVER:
          break
      ep_len = ep_offset_it - ep_offset

      data["ep_len"].append(ep_len)
      data["ep_offset"].append(ep_offset)


if __name__ == "__main__":
  main()
