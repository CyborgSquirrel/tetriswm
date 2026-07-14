import h5py
import torch
import numpy as np
from pathlib import Path
from typing import Callable, Optional
from einops import rearrange
from .traj_dset import TrajDataset, get_train_val_sliced


class TetrisDataset(TrajDataset):
    def __init__(
        self,
        data_path: str,
        n_rollout: Optional[int] = None,
        transform: Optional[Callable] = None,
        normalize_action: bool = False,
    ):
        self.data_path = Path(data_path)
        self.transform = transform

        with h5py.File(self.data_path, "r") as f:
            ep_len = f["ep_len"][:]
            ep_offset = f["ep_offset"][:]
            actions = f["action"][:].astype(np.float32)
            states = f["state_placed_grid"][:].astype(np.float32)  # (N, W, H) uint8

        n = n_rollout if n_rollout is not None else len(ep_len)
        self.ep_len = torch.from_numpy(ep_len[:n]).long()
        self.ep_offset = torch.from_numpy(ep_offset[:n]).long()

        # Flatten the 2D grid into a 1D proprioceptive vector
        states = rearrange(torch.from_numpy(states), "n w h -> n (w h)")

        max_end = (self.ep_offset[-1] + self.ep_len[-1]).item()
        self.actions_flat = torch.from_numpy(actions[:max_end])
        self.states_flat = states[:max_end]

        self.action_dim = self.actions_flat.shape[-1]
        self.state_dim = self.states_flat.shape[-1]
        self.proprio_dim = self.states_flat.shape[-1]

        if normalize_action:
            self.action_mean = self.actions_flat.mean(0)
            self.action_std = self.actions_flat.std(0).clamp(min=1e-6)
        else:
            self.action_mean = torch.zeros(self.action_dim)
            self.action_std = torch.ones(self.action_dim)

        self.state_mean = torch.zeros(self.state_dim)
        self.state_std = torch.ones(self.state_dim)
        self.proprio_mean = torch.zeros(self.proprio_dim)
        self.proprio_std = torch.ones(self.proprio_dim)

        print(f"Loaded {n} tetris episodes from {self.data_path}")

    def get_seq_length(self, idx: int) -> int:
        return self.ep_len[idx].item()

    def __len__(self) -> int:
        return len(self.ep_len)

    def get_all_actions(self):
        parts = []
        for i in range(len(self.ep_len)):
            o = self.ep_offset[i].item()
            t = self.ep_len[i].item()
            parts.append(self.actions_flat[o : o + t])
        return torch.cat(parts, dim=0)

    def get_frames(self, idx: int, frames):
        offset = self.ep_offset[idx].item()
        frames = list(frames)
        abs_idx = [offset + f for f in frames]

        with h5py.File(self.data_path, "r") as f:
            # Episodes are contiguous — read the full episode slice then index
            ep_start = offset
            ep_end = offset + self.ep_len[idx].item()
            pixels = f["pixels"][ep_start:ep_end][frames]  # (T, H, W, C)

        pixels = torch.from_numpy(pixels.astype(np.float32)) / 255.0
        pixels = rearrange(pixels, "t h w c -> t c h w")
        if self.transform:
            pixels = self.transform(pixels)

        act = (self.actions_flat[abs_idx] - self.action_mean) / self.action_std
        state = self.states_flat[abs_idx]
        proprio = state

        obs = {"visual": pixels, "proprio": proprio}
        return obs, act, state, {}

    def __getitem__(self, idx: int):
        return self.get_frames(idx, range(self.get_seq_length(idx)))

    def preprocess_imgs(self, imgs):
        if isinstance(imgs, torch.Tensor):
            return rearrange(imgs, "b h w c -> b c h w") / 255.0
        raise NotImplementedError


def load_tetris_slice_train_val(
    transform,
    data_path: str,
    n_rollout: Optional[int] = None,
    normalize_action: bool = False,
    split_ratio: float = 0.9,
    num_hist: int = 0,
    num_pred: int = 0,
    frameskip: int = 1,
):
    dset = TetrisDataset(
        data_path=data_path,
        n_rollout=n_rollout,
        transform=transform,
        normalize_action=normalize_action,
    )
    dset_train, dset_val, train_slices, val_slices = get_train_val_sliced(
        traj_dataset=dset,
        train_fraction=split_ratio,
        num_frames=num_hist + num_pred,
        frameskip=frameskip,
    )
    return {"train": train_slices, "valid": val_slices}, {"train": dset_train, "valid": dset_val}
