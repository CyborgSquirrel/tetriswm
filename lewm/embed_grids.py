"""Get WM embeddings for an empty tetris grid and n dataset-sampled grids, save to HDF5.

Index 0 is the empty grid; indices 1..n are sampled from the dataset.
"""

import sys
import contextlib as ctl
import os.path
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import hydra
import h5py
import numpy as np
import torch
from einops import rearrange
from omegaconf import DictConfig
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

import lewm.jepa
import lewm.module
import lewm.patch_jepa
import lewm.utils

sys.modules.setdefault("jepa", lewm.jepa)
sys.modules.setdefault("patch_jepa", lewm.patch_jepa)
sys.modules.setdefault("module", lewm.module)

import stable_worldmodel as swm
from tetris.tetris import Tetris
from utils import LazyDataset


def load_model(path, device):
    model = torch.load(path, weights_only=False, map_location=device)
    if hasattr(model, "model"):
        model = model.model
    return model.eval().requires_grad_(False)


def render_empty(width, height, render_size):
    tetris = Tetris(width=width, height=height, render_size=render_size)
    tetris.grid = np.zeros((width, height), dtype=np.uint8)
    return tetris.render()


def encode_batch(model, pixels_batch, img_size):
    # pixels_batch: (B, 1, C, H, W) from DataLoader
    transform = lewm.utils.get_img_preprocessor(source="pixels", target="pixels", img_size=img_size)
    info = {"pixels": pixels_batch}
    info = transform(info)
    with torch.no_grad():
        model.encode(info)
    return info["emb"].squeeze(1).numpy()  # (B, D)


def encode_single(model, pixels, img_size):
    pixels_t = torch.from_numpy(pixels)
    pixels_t = rearrange(pixels_t, "h w c -> 1 1 c h w")
    return encode_batch(model, pixels_t, img_size).squeeze(0)  # (D,)


@hydra.main(config_path="../conf", config_name="embed_grids", version_base=None)
def main(cfg: DictConfig):
    rng = np.random.default_rng(cfg.seed)
    model = load_model(cfg.checkpoint, cfg.device)

    dataset = swm.data.HDF5Dataset(
        name=cfg.dataset.name,
        cache_dir=cfg.dataset.cache_dir,
        num_steps=1,
        frameskip=1,
        keys_to_load=["pixels", "tetris_grid_mask"],
    )

    # n = min(cfg.n, len(dataset))
    # indices = rng.choice(len(dataset), size=n, replace=False)
    # indices.sort()
    # dataset = Subset(dataset, indices.tolist())
    loader = DataLoader(dataset, batch_size=cfg.batch_size, shuffle=False, num_workers=0)

    # Determine img_size from first batch
    first_batch = next(iter(loader))
    # pixels shape: (B, 1, C, H, W)
    img_size = first_batch["pixels"].shape[-1]

    # Determine grid dimensions from first mask: (B, 1, W, H) or (B, 1, H, W)
    first_mask = first_batch["tetris_grid_mask"]  # (B, 1, W, H)
    grid_w, grid_h = first_mask.shape[-2], first_mask.shape[-1]

    output = os.path.join(os.path.subdir(cfg.checkpoint), "embeddings.h5")

    with ctl.ExitStack() as stack:
        f = stack.enter_context(h5py.File(output, "w"))
        emb_ds = stack.enter_context(LazyDataset(f, "emb"))
        mask_ds = stack.enter_context(LazyDataset(f, "mask"))
        idx_ds = stack.enter_context(LazyDataset(f, "idx"))

        # index 0: empty grid
        # empty_pixels = render_empty(grid_w, grid_h, img_size)
        # emb_ds.append(encode_single(model, empty_pixels, img_size))
        # mask_ds.append(np.zeros((grid_w, grid_h), dtype=np.uint8))
        # idx_ds.append(np.int64(-1))

        # indices 1..n: sampled from dataset
        src_idx = 0
        for batch in tqdm(loader, total=len(loader)):
            pixels = batch["pixels"]              # (B, 1, C, H, W)
            masks = batch["tetris_grid_mask"]     # (B, 1, W, H)
            embs = encode_batch(model, pixels, img_size)  # (B, D)
            B = embs.shape[0]
            for i in range(B):
                emb_ds.append(embs[i])
                mask_ds.append(masks[i, 0].numpy())
                idx_ds.append(np.int64(indices[src_idx]))
                src_idx += 1

    print(f"Saved {n} embeddings to {repr(output)}")


if __name__ == "__main__":
    main()
