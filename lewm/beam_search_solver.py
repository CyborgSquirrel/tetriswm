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

    Encodes pixels and goal once per solve call, then at each depth appends
    one action embedding and runs a single predict step rather than re-rolling
    the full trajectory from scratch.  Cost is O(beam_width * n_actions * depth)
    predictor steps total.

    Args:
        model: JEPA world model with encode_pixels, action_trans, action_encoder, predict.
        beam_width: Maximum number of beams kept after each pruning step.
        device: Device for tensor computations.
        seed: Unused; kept for API parity with other solvers.
    """

    def __init__(
        self,
        model: Costable,
        beam_width: int = 10,
        batch_size: int = 1,
        device: str | torch.device = "cpu",
        seed: int = 1234,
        lazy: bool = True,
        total_steps = None,
        scorer=None,
    ) -> None:
        self.model = model.to(device)
        self.beam_width = beam_width
        self.batch_size = batch_size
        self.device = device

        self._n_envs = None
        self._n_actions = None
        self._config = None
        self._configured = False

        self.lazy = lazy
        # Optional callable (bs, k, D) -> (bs, k); if set, replaces goal-MSE cost.
        self.scorer = scorer

        self.total_steps = total_steps
        self.step_idx = 0

    def configure(self, *, action_space: gym.Space, n_envs: int, config: Any) -> None:
        self._n_envs = n_envs
        self._config = config
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

        Encodes pixels and goal once, then incrementally predicts one new
        embedding per depth level rather than re-rolling from scratch.

        Args:
            info_dict: Environment state dict with 'pixels' and 'goal' keys.
            init_action: Ignored (beam search does not warm-start).

        Returns:
            dict with 'actions' key of shape (n_envs, receding_horizon, n_actions).
        """

        if self.total_steps is not None:
            prev_step_idx, self.step_idx = self.step_idx, self.step_idx + self._config.receding_horizon
            plan_step_idx = min(self.step_idx + self._config.horizon - self._config.receding_horizon, self.total_steps-1)
            plan_steps = plan_step_idx - prev_step_idx
        else:
            plan_steps = self.horizon

        n_envs = self._n_envs
        n_actions = self._n_actions
        beam_width = self.beam_width
        device = self.device
        model = self.model
        HS = getattr(model, "history_size", 3)

        # Precompute action embeddings for all one-hot actions: (n_actions, A_emb)
        all_act_oh = torch.eye(n_actions, device=device)
        all_trans = model.action_trans(all_act_oh.unsqueeze(0))   # (1, n_actions, action_dim)
        all_act_emb = model.action_encoder(all_trans).squeeze(0)  # (n_actions, A_emb)
        A_emb = all_act_emb.shape[-1]

        start_time = time.time()
        all_best = []

        for start in range(0, n_envs, self.batch_size):
            end = min(start + self.batch_size, n_envs)
            bs = end - start

            info_b = {k: (v[start:end].to(device) if torch.is_tensor(v) else v)
                      for k, v in info_dict.items()}

            # Encode initial pixels once: (bs, T_init, D)
            init_pix = info_b["pixels"].to(device)
            init_emb = model.encode_pixels(init_pix)
            D_emb = init_emb.shape[-1]

            if self.scorer is None:
                # Encode goal once.
                # goal is (bs, H, W, C) — no T dim; add it so encode_pixels sees (bs, 1, H, W, C).
                goal_pix = info_b["goal"].to(device)
                goal_emb = model.encode_pixels(goal_pix)  # (bs, 1, D)
                goal_last = goal_emb[:, -1]  # (bs, D)

            # k - number of trajectories kept
            k = 1

            # Beam state:
            #   beam_emb_hist:     (bs, k, T_init, D)    -- grows by 1 per depth
            #   beam_act_emb_hist: (bs, k, 0, A_emb)     -- grows by 1 per depth
            #   beam_act_seqs:     (bs, k, 0, n_actions)  -- one-hot action history
            beam_emb_hist = rearrange(init_emb, "b 1 d -> b 1 1 d")
            beam_act_emb_hist = torch.zeros(bs, 1, 0, A_emb, device=device)
            beam_act_seqs     = torch.zeros(bs, 1, 0, n_actions, device=device)
            if self.lazy:
                beam_min_cost = torch.full((bs, 1), torch.inf, device=device)

            k = 1

            for depth in range(1, plan_steps + 1):
                # n_beams = beam_emb_hist.shape[1]
                # n_cands = n_beams * n_actions

                # Expand each beam by n_actions
                emb_exp     = repeat(beam_emb_hist    , "b k t d -> b (k a) t d", a=n_actions)
                act_emb_exp = repeat(beam_act_emb_hist, "b k t d -> b (k a) t d", a=n_actions)
                act_seq_exp = repeat(beam_act_seqs    , "b k t d -> b (k a) t d", a=n_actions)
                if self.lazy:
                    min_cost_exp = repeat(beam_min_cost, "b k -> b (k a)", a=n_actions)

                # Append new action embedding and one-hot for each candidate
                new_act_emb = repeat(all_act_emb, "a d -> b (k a) 1 d", b=bs, k=k)
                new_act_oh  = repeat(all_act_oh, "a d -> b (k a) 1 d", b=bs, k=k)

                k *= n_actions

                act_emb_full = torch.cat([act_emb_exp, new_act_emb], dim=2)  # (bs, k, depth, A_emb)
                act_seq_full = torch.cat([act_seq_exp, new_act_oh ], dim=2)   # (bs, k, depth, n_actions)

                # Single predict step for all candidates
                emb_exp_flat = rearrange(emb_exp, "b k t d -> (b k) t d")
                act_emb_full_flat = rearrange(act_emb_full, "b k t d -> (b k) t d")

                # (bs, k*n_cands, 1, D)
                next_emb_flat = model.predict(
                    emb_exp_flat     [:, -HS:],
                    act_emb_full_flat[:, -HS:],
                )[:, -1:]

                next_emb = rearrange(next_emb_flat, "(b k) t d -> b k t d", b=bs, k=k)

                # Cost: either learned scorer or MSE against goal embedding
                if self.scorer is not None:
                    costs = -self.scorer(next_emb[:, :, -1, :])  # (bs, k)
                else:
                    costs = (next_emb - rearrange(goal_last, "b d -> b 1 1 d")).pow(2).sum((-1, -2))  # (bs, k)
                if self.lazy:
                    costs = torch.minimum(min_cost_exp, costs)
                    min_cost_exp = costs

                # Prune to top-K beams
                k = min(beam_width, k)
                _, topk_idx = costs.topk(k, dim=1, largest=False)
                b_idx = torch.arange(bs)[..., None]

                # Gather surviving emb hist and append predicted next embedding
                surv_emb = emb_exp[b_idx, topk_idx]
                surv_next = next_emb[b_idx, topk_idx]
                beam_emb_hist = torch.cat([surv_emb, surv_next], dim=2)[:, :, -HS:]

                # Gather surviving act emb hist and action sequences
                beam_act_emb_hist = act_emb_full[b_idx, topk_idx][:, :, -HS:]
                beam_act_seqs = act_seq_full[b_idx, topk_idx]

                if self.lazy:
                    beam_min_cost = min_cost_exp[b_idx, topk_idx]

            # Best action sequence for this batch: (bs, k, plan_depth, n_actions)
            all_best.append(beam_act_seqs[:, 0].cpu())

        best_seqs = torch.cat(all_best, dim=0)

        # print(f"BeamSearchSolver.solve {time.time() - start_time:.3f}s "
        #       f"(beam_width={beam_width}, depth={plan_steps})")

        return {"actions": best_seqs}
