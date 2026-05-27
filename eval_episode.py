#!/usr/bin/env python3
"""
Evaluate a world model on a single episode from a dataset.

Usage:
    python eval_episode.py \\
        eval.checkpoint=outputs/.../model_epoch_100_object.ckpt \\
        eval.dataset_path=/path/to/data.h5 \\
        eval.episode_idx=0 \\
        data=tetris
"""

from pathlib import Path
from collections import deque as Deque
import contextlib as ctl
import os
import itertools as itt
import sys

import numpy as np
from omegaconf import open_dict
import h5py
import hydra
import torch
from einops import rearrange, pack

# sys.path.insert(0, str(Path(__file__).parent))

import stable_pretraining as spt
import stable_worldmodel as swm
from lewm.utils import get_img_preprocessor, agg
from tetris.tetris import Action, Tetris
from utils import LazyDataset, Tee


def load_world_model(path):
    model = torch.load(Path(path), map_location="cpu", weights_only=False)
    if hasattr(model, "model"):
        model = model.model
    return model.eval().requires_grad_(False)


def build_dataset(cfg):
    dataset = swm.data.HDF5Dataset(
        cfg.eval.dataset_path,
        num_steps=1,
        frameskip=cfg.data.dataset.frameskip,
        keys_to_load=["pixels", "action"],
        keys_to_cache=["action"],
        transform=None,
    )
    transform = get_img_preprocessor(source="pixels", target="pixels", img_size=cfg.img_size)
    dataset.transform = spt.data.transforms.Compose(transform)
    return dataset


def load_episode(dataset, episode_idx):
    num_episodes = len(dataset.lengths)
    if episode_idx >= num_episodes:
        raise ValueError(f"episode_idx={episode_idx} out of range (dataset has {num_episodes} episodes)")
    episode = dataset.load_episode(episode_idx)
    episode["action"] = episode["action"].reshape(-1, episode["action"].shape[-1])
    return episode


@hydra.main(version_base=None, config_path="./conf", config_name="eval_episode")
def run(cfg):
    path = Path.home() / ".stable_worldmodel" / cfg.subdir
    path.mkdir(parents=True, exist_ok=True)

    lf = (path / "log.txt").open("w")
    sys.stdout = Tee(sys.stdout, lf)
    sys.stderr = Tee(sys.stderr, lf)

    print("Saving results to {cfg.subdir}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"Checkpoint   : {cfg.eval.checkpoint}")
    print(f"Dataset      : {cfg.eval.dataset_path}")
    print(f"Episode idx  : {cfg.eval.episode_idx}")

    model = load_world_model(cfg.eval.checkpoint)
    dataset = build_dataset(cfg)
    episode = load_episode(dataset, cfg.eval.episode_idx)

    T = episode["pixels"].shape[0]
    print(f"Episode {cfg.eval.episode_idx}: {T} steps, pixels {tuple(episode['pixels'].shape)}")

    # NOTE: assuming frameskip=1 because big lazy
    if cfg.data.dataset.frameskip != 1:
        raise RuntimeError(f"Can't handle {cfg.data.dataset.frameskip=}")

    with open_dict(cfg):
        cfg.wm.action_dim = 7

    actions_dict = {
        Action.NOTHING     : [1,0,0,0,0,0,0],
        Action.MOVE_LEFT   : [0,1,0,0,0,0,0],
        Action.MOVE_RIGHT  : [0,0,1,0,0,0,0],
        Action.MOVE_DOWN   : [0,0,0,1,0,0,0],
        Action.HARD_DROP   : [0,0,0,0,1,0,0],
        Action.ROTATE_LEFT : [0,0,0,0,0,1,0],
        Action.ROTATE_RIGHT: [0,0,0,0,0,0,1],
    }
    actions_dict = {
        k: torch.tensor(v, dtype=torch.float64) for k,v in actions_dict.items()
    }

    def convert_onehot_action(query_action_onehot):
        for action, action_onehot in actions_dict.items():
            if torch.all(torch.isclose(query_action_onehot, action_onehot)):
                return action
        raise RuntimeError()

    # Actual evaluation
    emb_hist = Deque(maxlen=cfg.wm.history_size)
    emb_hist.append(model.encode_pixels(episode["pixels"][None, None, 0]))
    gt_act_hist = []
    act_hist = []
    act_match_hist = []
    for i in range(T-1):
        enc_emb_1 = model.encode_pixels(episode["pixels"][None, None, i+1])

        action = episode["action"][i]
        gt_act_hist.append(action)
        act_hist.append(action)
        act = rearrange(act_hist[-cfg.wm.history_size:], "t fs_d -> 1 t fs_d")

        print("GT action", action)

        emb_0 = rearrange(list(emb_hist), "t 1 1 d -> 1 t d")
        with agg.min(key=lambda a: a[0]) as best:
            for last_act in actions_dict.values():
                act_mod = act.clone()
                act_mod[:, -1, -cfg.wm.action_dim:] = last_act
                act_mod_emb = model.action_encoder(model.action_trans(act_mod))

                pred_emb_1 = model.predict(emb_0, act_mod_emb)[:, (-1,), :]
                l2 = torch.mean((enc_emb_1 - pred_emb_1) ** 2).item()
                best.update((l2, pred_emb_1, last_act))

                print(f"  Probe action ({l2:.5f})", last_act)

        best_l2 = best.result[0]
        best_emb = best.result[1]
        best_action = best.result[2]
        act_match = torch.all(torch.isclose(action, best_action)).item()

        print(f"Best action ({best_l2:.5f}), {act_match=}", best_action)

        emb_hist.append(best_emb)
        act_hist[-1] = best_action

        act_match_hist.append(act_match)

    act_match_hist = np.array(act_match_hist)
    act_match_prop = np.sum(act_match_hist) / len(act_match_hist)
    print(f"Percentage of matching actions taken {act_match_prop*100:.2f}%")

    with ctl.ExitStack() as stack:
        f = stack.enter_context(h5py.File(path / "results.h5", "w"))

        def simulate(actions, f_pixels, f_action):
            tetris = Tetris.from_config(cfg.tetris.tetris)
            for act_onehot in itt.chain(actions, (None,)):
                f_pixels.append(tetris.render())

                act_append = act_onehot
                if act_append is None:
                    act_append = torch.full(f_action.dataset.shape[1:], 0, dtype=torch.float32)
                f_action.append(act_append.numpy())

                if act_onehot is not None:
                    act = convert_onehot_action(act_onehot)
                    tetris.step(act)

        f["action_match"] = act_match_hist

        simulate(
            gt_act_hist,
            stack.enter_context(LazyDataset(f, "gt_pixels")),
            stack.enter_context(LazyDataset(f, "gt_action")),
        )
        simulate(
            act_hist,
            stack.enter_context(LazyDataset(f, "pd_pixels")),
            stack.enter_context(LazyDataset(f, "pd_action")),
        )


if __name__ == "__main__":
    run()
