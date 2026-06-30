"""CMA-ES optimization of a linear scoring function for beam-search Tetris play.

The scorer is a 192->1 linear layer (193 params total). Beam search uses it to
rank WM predicted embeddings; CMA-ES maximizes lines cleared in the real env.
"""

import os
import sys
import math
from pathlib import Path

os.environ["MUJOCO_GL"] = "egl"
sys.path.insert(0, str(Path(__file__).parent.parent))

import cma
import numpy as np
import torch
import hydra
import wandb
import stable_pretraining as spt
import stable_worldmodel as swm
from omegaconf import DictConfig, OmegaConf

import tetris.gym_env  # noqa: F401
from tetris.gym_env import TetrisEnv
from torchvision.transforms import v2 as transforms

from lewm.beam_search_solver import BeamSearchSolver


def load_world_model(path: str):
    model = torch.load(path, map_location="cpu", weights_only=False)
    if hasattr(model, "model"):
        model = model.model
    return model.eval().requires_grad_(False)


def make_transform(render_size: int):
    return transforms.Compose([
        transforms.ToImage(),
        transforms.ToDtype(torch.float32, scale=True),
        transforms.Normalize(**spt.data.dataset_stats.ImageNet),
    ])


def make_scorer(params: np.ndarray, emb_dim: int, device):
    """Build a scorer callable from a flat parameter vector (w, b)."""
    w = torch.tensor(params[:emb_dim], dtype=torch.float32, device=device)
    b = torch.tensor(params[emb_dim],  dtype=torch.float32, device=device)

    @torch.inference_mode()
    def scorer(emb):  # (bs, k, D) -> (bs, k)
        return emb @ w + b

    return scorer


class Fitness:
    def update(self, env):
        pass

    def finish(self, env):
        pass


class ClearedRowsFitness(Fitness):
    def __init__(self):
        self.value = 0

    def finish(self, env):
        self.value = env._tetris.dynamics.cleared_rows


class CleanFitness(Fitness):
    def __init__(self):
        self.value = 0

    def update(self, env):
        total_tiles = math.prod(env._tetris.grid.shape)
        empty_tiles = np.sum(env._tetris.grid == 0)
        empty_prop = empty_tiles / total_tiles
        self.value += empty_prop ** 2

    def finish(self, env):
        pass


def run_episode(env, solver, transform, max_steps, fitness, device):
    env.reset()
    frames = []
    for _ in range(max_steps):
        for f in fitness:
            f.update(env)

        frame = env.render()
        frames.append(frame)
        pixels = transform(frame).unsqueeze(0).unsqueeze(0).to(device)  # (1,1,C,H,W)
        result = solver.solve({"pixels": pixels})
        action = result["actions"][0, 0].numpy()
        _, _, terminated, truncated, _ = env.step(action)
        if terminated or truncated:
            break

    for f in fitness:
        f.update(env)
        f.finish(env)

    return frames


def evaluate(params, emb_dim, solver, env, transform, max_steps, n_episodes, device, video_dir, tag, consistency_lambda, fitness_cfg):
    import imageio.v2 as imageio
    solver.scorer = make_scorer(params, emb_dim, device)

    all_fit = []
    all_lines = []

    for ep in range(n_episodes):
        fitness_fn = hydra.utils.instantiate(fitness_cfg)
        frames = run_episode(env, solver, transform, max_steps, [fitness_fn], device)
        all_fit.append(fitness_fn.value)
        all_lines.append(env._tetris.dynamics.cleared_rows)

        video_path = video_dir / f"{tag}_ep{ep}.mp4"
        imageio.mimwrite(str(video_path), frames, fps=10)

    # Lines stats
    mean_lines = np.mean(all_lines).item()
    std_lines = np.std(all_lines).item()

    # Fitness stats + aggregation
    mean_fit = np.mean(all_fit).item()
    std_fit = np.std(all_fit).item()
    fit = mean_fit - consistency_lambda * std_fit

    return fit, mean_fit, std_fit, mean_lines, std_lines


@hydra.main(version_base=None, config_path="../conf", config_name="cmaes_tetris")
def run(cfg: DictConfig):
    device = torch.device(cfg.device if torch.cuda.is_available() else "cpu")

    model = load_world_model(cfg.checkpoint)
    model = model.to(device)
    model.interpolate_pos_encoding = True

    transform = make_transform(cfg.tetris_cfg.render_size)
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

    emb_dim = cfg.emb_dim
    n_params = emb_dim + 1

    es = cma.CMAEvolutionStrategy(
        np.zeros(n_params),
        cfg.sigma0,
        {"maxiter": cfg.max_iter, "tolx": 1e-6, "tolfun": 1e-6, "verbose": 3, "seed": cfg.seed},
    )

    if (p := OmegaConf.select(cfg, "subdir")) is not None:
        subdir = Path(swm.data.utils.get_cache_dir(), p)
        subdir.mkdir(parents=True, exist_ok=True)
    else:
        subdir = Path(".")

    if cfg.wandb.enabled:
        wandb.init(**cfg.wandb.config, config=OmegaConf.to_container(cfg, resolve=True))

    best_fit = -np.inf
    best_params = None
    iteration = 0

    while not es.stop():
        solutions = es.ask()
        results = [
            evaluate(
                x, emb_dim, solver, env,
                transform, cfg.max_steps, cfg.n_eval_episodes, device,
                subdir, tag=f"iter{iteration:04d}_cand{i:03d}",
                consistency_lambda=cfg.consistency_lambda,
                fitness_cfg=cfg.fitness,
            )
            for i, x in enumerate(solutions)
        ]
        fit, mean_fit, std_fit, mean_lines, std_lines = zip(*results)
        fit = list(fit)

        # CMA-ES minimizes, so negate fitness
        es.tell(solutions, [-f for f in fit])
        es.disp()

        best_idx = int(np.argmax(fit))
        gen_best = fit[best_idx]
        if gen_best > best_fit:
            best_fit = gen_best
            best_params = solutions[best_idx].copy()
            print(f"New best: {best_fit:.2f} (fitness val: {mean_fit[best_idx]:.2f}±{std_fit[best_idx]:.2f}, lines: {mean_lines[best_idx]:.2f}±{std_lines[best_idx]:.2f})")

        if cfg.wandb.enabled:
            best_video_tag = f"iter{iteration:04d}_cand{best_idx:03d}_ep0"
            wandb.log({
                "fitness/best": gen_best,
                "fitness/mean": np.mean(fit).item(),
                "fitness/min": np.min(fit).item(),
                "fitness/all_time_best": best_fit,
                "lines/best_mean": mean_lines[best_idx],
                "lines/best_std": std_lines[best_idx],
                "lines/pop_mean": np.mean(mean_lines).item(),
                "cmaes/sigma": es.sigma,
                "best_episode": wandb.Video(str(subdir / f"{best_video_tag}.mp4"), fps=10),
            }, step=iteration)

        iteration += 1
        if best_params is not None and iteration % cfg.checkpoint_interval == 0:
            ckpt_path = subdir / f"cmaes_scorer_iter{iteration:04d}.npy"
            np.save(ckpt_path, best_params)
            print(f"Checkpoint saved to {ckpt_path}")

    print(f"\nOptimization complete. Best score: {best_fit:.2f} lines/episode")
    out_path = subdir / "cmaes_scorer.npy"
    np.save(out_path, best_params)
    print(f"Best params saved to {out_path}")

    if cfg.wandb.enabled:
        wandb.finish()


if __name__ == "__main__":
    run()
