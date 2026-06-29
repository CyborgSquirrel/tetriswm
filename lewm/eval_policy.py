"""Evaluate a CMA-ES trained linear scorer on the real Tetris environment.

Loads a flat parameter vector (.npy) produced by cmaes_tetris.py and runs
n_episodes, reporting average lines cleared.
"""

import os
import sys
import contextlib as ctl
from pathlib import Path

os.environ["MUJOCO_GL"] = "egl"
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import hydra
import torch
import stable_pretraining as spt
import stable_worldmodel as swm
from omegaconf import DictConfig, OmegaConf
from torchvision.transforms import v2 as transforms

import tetris.gym_env  # noqa: F401
from tetris.gym_env import TetrisEnv

from lewm.beam_search_solver import BeamSearchSolver
from utils import TeeStdout, TeeStderr


def load_world_model(path: str):
    model = torch.load(path, map_location="cpu", weights_only=False)
    if hasattr(model, "model"):
        model = model.model
    return model.eval().requires_grad_(False)


def make_transform():
    return transforms.Compose([
        transforms.ToImage(),
        transforms.ToDtype(torch.float32, scale=True),
        transforms.Normalize(**spt.data.dataset_stats.ImageNet),
    ])


def make_scorer(params: np.ndarray, emb_dim: int, device):
    w = torch.tensor(params[:emb_dim], dtype=torch.float32, device=device)
    b = torch.tensor(params[emb_dim], dtype=torch.float32, device=device)

    @torch.inference_mode()
    def scorer(emb):  # (bs, k, D) -> (bs, k)
        return emb @ w + b

    return scorer


def run_episode(env, solver, transform, max_steps, device, save_video):
    env.reset()
    frames = []
    for _ in range(max_steps):
        frame = env.render()
        if save_video:
            frames.append(frame)
        pixels = transform(frame).unsqueeze(0).unsqueeze(0).to(device)  # (1,1,C,H,W)
        result = solver.solve({"pixels": pixels})
        action = result["actions"][0, 0].numpy()
        _, _, terminated, truncated, _ = env.step(action)
        if terminated or truncated:
            break
    return env._tetris.dynamics.cleared_rows, frames


@hydra.main(version_base=None, config_path="../conf", config_name="eval_policy")
def run(cfg: DictConfig):
    device = torch.device(cfg.device if torch.cuda.is_available() else "cpu")

    model = load_world_model(cfg.checkpoint)
    model = model.to(device)
    model.interpolate_pos_encoding = True

    transform = make_transform()
    plan_config = swm.PlanConfig(**cfg.plan_config)

    env = TetrisEnv(cfg.tetris_cfg)
    solver = BeamSearchSolver(
        model=model,
        beam_width=cfg.beam_width,
        batch_size=1,
        device=device,
        lazy=cfg.lazy,
    )
    solver.configure(action_space=env.action_space, n_envs=1, config=plan_config)

    params = np.load(cfg.policy)
    emb_dim = cfg.emb_dim
    solver.scorer = make_scorer(params, emb_dim, device)

    subdir = None
    if (p := OmegaConf.select(cfg, "subdir")) is not None:
        subdir = Path(swm.data.utils.get_cache_dir(), p)
        subdir.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(cfg.seed)
    all_lines = []

    for ep in range(cfg.n_episodes):
        seed = int(rng.integers(0, 2**31))
        env.reset(seed=seed)
        lines, frames = run_episode(env, solver, transform, cfg.max_steps, device, save_video=subdir is not None)
        all_lines.append(lines)
        print(f"Episode {ep + 1}/{cfg.n_episodes}: {lines} lines cleared")

        if subdir is not None and frames:
            import imageio.v2 as imageio
            video_path = subdir / f"episode_{ep:03d}.mp4"
            imageio.mimwrite(str(video_path), frames, fps=10)

    with ctl.ExitStack() as stack:
        if subdir is not None:
            lf = stack.enter_context((subdir / "log.txt").open("w"))
            stack.enter_context(TeeStdout(lf))
            stack.enter_context(TeeStderr(lf))

        print("=== Config ===")
        print(OmegaConf.to_yaml(cfg, resolve=True))

        print("=== Results ===")
        print(f"Episodes: {cfg.n_episodes}")
        print(f"Lines cleared: {all_lines}")
        print(f"Lines cleared — mean: {np.mean(all_lines):.2f}, "
              f"min: {np.min(all_lines)}, max: {np.max(all_lines)}")


if __name__ == "__main__":
    run()
