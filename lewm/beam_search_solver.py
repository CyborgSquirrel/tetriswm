"""Beam search solver for discrete action spaces."""

import time
from typing import Any

import gymnasium as gym
import numpy as np
import torch
from einops import rearrange, repeat

from stable_worldmodel.solver.solver import Costable


class BeamSearchSolver:
    """Beam search planner for discrete action spaces.

    At each depth, every active beam entry is expanded with all possible
    actions, scored via the world model, and pruned to the top-K beams.
    The encoder runs once per solve call (shared across all beam entries),
    so cost scales as O(beam_width * n_actions * horizon) predictor steps.

    Args:
        model: World model implementing the Costable protocol (needs get_cost).
        beam_width: Maximum number of beams to keep after each pruning step.
        device: Device for tensor computations.
        seed: Unused; kept for API parity with other solvers.
    """

    def __init__(
        self,
        model: Costable,
        total_steps: int,
        beam_width: int = 10,
        batch_size: int = 1,
        device: str | torch.device = "cpu",
        seed: int = 1234,
    ) -> None:
        self.model = model
        self.total_steps = total_steps
        self.beam_width = beam_width
        self.batch_size = batch_size
        self.device = device

        self._n_envs = None
        self._n_actions = None
        self._config = None
        self._configured = False

        self.step_idx = 0

    def configure(self, *, action_space: gym.Space, n_envs: int, config: Any) -> None:
        self._n_envs = n_envs
        self._config = config
        print(self._config)
        # Support both Discrete and custom OneHotSpace
        self._n_actions = getattr(action_space, "n", int(np.prod(action_space.shape)))
        self._configured = True

    @property
    def n_envs(self) -> int:
        return self._n_envs

    @property
    def action_dim(self) -> int:
        return self._n_actions * self._config.action_block

    @property
    def horizon(self) -> int:
        return self._config.horizon

    def __call__(self, *args: Any, **kwargs: Any) -> dict:
        return self.solve(*args, **kwargs)

    @torch.inference_mode()
    def solve(self, info_dict: dict, init_action: torch.Tensor | None = None) -> dict:
        """Beam search over action sequences, scored by world model cost.

        Args:
            info_dict: Environment state dict with at least 'pixels' and 'goal'.
            init_action: Ignored (beam search does not warm-start).

        Returns:
            dict with 'actions' key of shape (n_envs, horizon, n_actions).
        """

        prev_step_idx, self.step_idx = self.step_idx, self.step_idx + self._config.receding_horizon
        plan_step_idx = min(self.step_idx + self._config.horizon - self._config.receding_horizon, self.total_steps-1)
        plan_steps = plan_step_idx - prev_step_idx

        start_time = time.time()

        n_envs = self._n_envs
        n_actions = self._n_actions
        horizon = self._config.horizon
        beam_width = self.beam_width
        device = self.device

        pixels = info_dict.get("pixels")
        # pixels shape from WorldModelPolicy: (n_envs, T_init, C, H, W)
        T_init = pixels.shape[1] if pixels is not None else 1

        # One-hot action vectors: (n_actions, n_actions)
        all_actions = torch.eye(n_actions, device=device)

        # Collect best sequences for each env: (n_envs, horizon, n_actions)
        all_best = []

        for start in range(0, n_envs, self.batch_size):
            end = min(start + self.batch_size, n_envs)
            bs = end - start  # current batch size

            # beam_seqs: (bs, cur_beam_size, depth, n_actions) — grows with depth
            beam_seqs: torch.Tensor | None = None

            for depth in range(1, plan_steps + 1):
                if beam_seqs is None:
                    # Depth 1: one candidate per action
                    n_cands = n_actions
                    # (n_actions, n_actions) → (bs, n_actions, 1, n_actions)
                    cand_seqs = repeat(all_actions, "a d -> b a 1 d", b=bs)
                else:
                    cur_beam_size = beam_seqs.shape[1]
                    n_cands = cur_beam_size * n_actions

                    # Tile each beam entry once per action
                    beam_expanded = repeat(beam_seqs, "b beam t d -> b beam a t d", a=n_actions)
                    # One-hot next step: (A, d) → (bs, B, A, 1, d)
                    next_acts = repeat(all_actions, "a d -> b beam a 1 d", b=bs, beam=cur_beam_size)

                    # Concat along time, fold (beam, action) → candidates
                    cand_seqs = rearrange(
                        torch.cat([beam_expanded, next_acts], dim=3),
                        "b beam a t d -> b (beam a) t d",
                    )

                # Prepend actual past actions as context: (bs, n_cands, T_init + depth, n_actions)
                # info_dict["action"] is (n_envs, T_init, n_actions) — the observed action history
                # past_acts = info_dict.get("action")
                # if past_acts is not None:
                #     ctx = repeat(past_acts[start:end].to(device), "b t d -> b s t d", s=n_cands)
                # else:
                #     ctx = torch.zeros(bs, n_cands, T_init, n_actions, device=device)
                # full_seqs = torch.cat([ctx, cand_seqs.to(device)], dim=2)

                # Expand info_dict: (bs, ...) → (bs, n_cands, ...)
                # Keeping bs=1 per batch means goal_emb stays [1, 1, D] so
                # expand_as inside criterion can broadcast correctly (same as CEM).
                info_exp: dict = {}
                for k, v in info_dict.items():
                    if torch.is_tensor(v):
                        info_exp[k] = repeat(v[start:end], "b ... -> b s ...", s=n_cands).to(device)
                    else:
                        info_exp[k] = v

                # Score all candidates; costs: (bs, n_cands)
                # print(full_seqs.shape)
                costs = self.model.get_cost(info_exp, cand_seqs)

                # Prune to top-K by ascending cost
                K = min(beam_width, n_cands)
                _, topk_idx = costs.topk(K, dim=1, largest=False)

                # Gather surviving sequences: (bs, K, depth, n_actions)
                idx_exp = repeat(topk_idx, "b k -> b k t d", t=depth, d=n_actions)
                beam_seqs = cand_seqs.to(device).gather(1, idx_exp)

            # beam_seqs: (bs, K, horizon, n_actions) — keep best per env
            all_best.append(beam_seqs[:, 0].cpu())

        # Concatenate across batches: (n_envs, horizon, n_actions)
        best_seqs = torch.cat(all_best, dim=0)

        print(f"BeamSearchSolver.solve completed in {time.time() - start_time:.4f}s  "
              f"(beam_width={beam_width}, horizon={horizon})")

        return {"actions": best_seqs.cpu()}
