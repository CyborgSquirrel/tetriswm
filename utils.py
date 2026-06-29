import logging
import random
import sys

import numpy as np
import h5py
from tqdm import tqdm
import hydra
from omegaconf import DictConfig

log = logging.getLogger(__name__)


class LazyDataset:
  def __init__(
    self,
    f: h5py.File,
    name: str,
    *,
    dataset_kwargs=None,
    init_hook=None,
  ):
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
    if not self.initted:
      log.error("Exited without initializing dataset %r", self.name)
      return
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


class Tee:
  def __init__(self, *files):
    self._files = files

  @property
  def files(self):
    yield from self._files

  def write(self, obj):
    for f in self.files:
      f.write(obj)
      f.flush()

  def flush(self):
    for f in self.files:
      f.flush()


class TeeStdout(Tee):
  def __enter__(self):
    sys.stdout, self.stdout = self, sys.stdout

  def __exit__(self, exc_type, exc_value, traceback):
    if sys.stdout is not self:
      raise RuntimeError("sys.stdout was changed")
    sys.stdout = self.stdout
    del self.stdout

  @property
  def files(self):
    yield self.stdout
    yield from self._files


class TeeStderr(Tee):
  def __enter__(self):
    sys.stderr, self.stderr = self, sys.stderr

  def __exit__(self, exc_type, exc_value, traceback):
    if sys.stderr is not self:
      raise RuntimeError("sys.stderr was changed")
    sys.stderr = self.stderr
    del self.stderr

  @property
  def files(self):
    yield self.stderr
    yield from self._files


def rng_derive_seed(rng: np.random.Generator | random.Random):
  if isinstance(rng, np.random.Generator):
    b = rng.bytes(4)
  elif isinstance(rng, random.Random):
    b = rng.randbytes(4)
  else:
    raise TypeError()
  seed = int.from_bytes(b)
  return seed
