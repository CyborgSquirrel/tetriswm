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
        raise NotImplementedError()

    def finish(self):
        raise NotImplementedError()


class CleanFitness(Fitness):
    def __init__(self):
        self.value = 0

    def update(self, env):
        total_tiles = math.prod(env._tetris.grid.shape)
        empty_tiles = np.sum(env._tetris.grid == 0)
        empty_prop = empty_tiles / total_tiles
        self.value += empty_prop ** 2

    def finish(self):
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
        f.finish()

    return env._tetris.dynamics.cleared_rows, frames


def evaluate(params, emb_dim, solver, env, transform, max_steps, n_episodes, device, video_dir, tag, consistency_lambda):
    import imageio.v2 as imageio
    solver.scorer = make_scorer(params, emb_dim, device)
    all_lines = []
    all_clean_fitness = []
    for ep in range(n_episodes):
        clean_fitness = CleanFitness()
        lines, frames = run_episode(env, solver, transform, max_steps, [clean_fitness], device)
        all_lines.append(lines)
        all_clean_fitness.append(clean_fitness.value)
        video_path = video_dir / f"{tag}_ep{ep}.mp4"
        imageio.mimwrite(str(video_path), frames, fps=10)
    mean_lines = float(np.mean(all_lines))
    std_lines = float(np.std(all_lines))
    # fitness = mean_lines - consistency_lambda * std_lines
    fitness = np.mean(all_clean_fitness)
    return fitness, mean_lines, std_lines


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

    best_score = -np.inf
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
            )
            for i, x in enumerate(solutions)
        ]
        fitnesses, mean_lines_all, std_lines_all = zip(*results)
        fitnesses = list(fitnesses)

        # CMA-ES minimizes, so negate fitness
        es.tell(solutions, [-f for f in fitnesses])
        es.disp()

        best_idx = int(np.argmax(fitnesses))
        gen_best = fitnesses[best_idx]
        if gen_best > best_score:
            best_score = gen_best
            best_params = solutions[best_idx].copy()
            print(f"New best: {best_score:.2f} (mean lines: {mean_lines_all[best_idx]:.2f}, std: {std_lines_all[best_idx]:.2f})")

        if cfg.wandb.enabled:
            best_video_tag = f"iter{iteration:04d}_cand{best_idx:03d}_ep0"
            wandb.log({
                "fitness/best": gen_best,
                "fitness/mean": float(np.mean(fitnesses)),
                "fitness/min": float(np.min(fitnesses)),
                "fitness/all_time_best": best_score,
                "lines/best_mean": mean_lines_all[best_idx],
                "lines/best_std": std_lines_all[best_idx],
                "lines/pop_mean": float(np.mean(mean_lines_all)),
                "cmaes/sigma": es.sigma,
                "best_episode": wandb.Video(str(subdir / f"{best_video_tag}.mp4"), fps=10),
            }, step=iteration)

        iteration += 1
        if best_params is not None and iteration % cfg.checkpoint_interval == 0:
            ckpt_path = subdir / f"cmaes_scorer_iter{iteration:04d}.npy"
            np.save(ckpt_path, best_params)
            print(f"Checkpoint saved to {ckpt_path}")

    print(f"\nOptimization complete. Best score: {best_score:.2f} lines/episode")
    out_path = subdir / "cmaes_scorer.npy"
    np.save(out_path, best_params)
    print(f"Best params saved to {out_path}")

    if cfg.wandb.enabled:
        wandb.finish()


if __name__ == "__main__":
    run()
