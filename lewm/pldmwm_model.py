"""PLDM World Model — IMPALA CNN encoder + GRU predictor.

Self-contained: no imports from the PLDM repo.
Implements the same interface as DinoWM so the BeamSearchSolver works unchanged.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange


# ---------------------------------------------------------------------------
# IMPALA CNN building blocks (adapted from /home/andrei/PLDM/pldm/models/encoders/impala.py)
# ---------------------------------------------------------------------------

class _ResnetBlock(nn.Module):
    def __init__(self, num_features):
        super().__init__()
        self.conv1 = nn.Conv2d(num_features, num_features, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(num_features, num_features, kernel_size=3, padding=1)

    def forward(self, x):
        identity = x
        out = F.relu(self.conv1(x))
        out = self.conv2(out)
        return F.relu(out + identity)


class _ResnetStack(nn.Module):
    def __init__(self, input_channels, num_features, num_blocks):
        super().__init__()
        self.initial_conv = nn.Conv2d(input_channels, num_features, kernel_size=3, padding=1)
        self.max_pool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.blocks = nn.ModuleList([_ResnetBlock(num_features) for _ in range(num_blocks)])

    def forward(self, x):
        x = self.initial_conv(x)
        x = self.max_pool(x)
        for block in self.blocks:
            x = block(x)
        return x


class ImpalaEncoder(nn.Module):
    """IMPALA CNN encoder.

    The MLP projection size is computed dynamically from img_size so the class
    works for any input resolution without hard-coding spatial dimensions.
    """

    def __init__(
        self,
        input_channels: int = 3,
        width: int = 1,
        stack_sizes: tuple = (16, 32, 32),
        num_blocks: int = 2,
        img_size: int = 84,
        embed_dim: int = 512,
        final_ln: bool = True,
    ):
        super().__init__()
        ch = [input_channels] + list(stack_sizes)
        self.stacks = nn.ModuleList([
            _ResnetStack(ch[i], s * width, num_blocks)
            for i, s in enumerate(stack_sizes)
        ])

        # Compute flat conv-output size via a dummy forward pass.
        with torch.no_grad():
            dummy = torch.zeros(1, input_channels, img_size, img_size)
            for stack in self.stacks:
                dummy = stack(dummy)
            dummy = F.relu(dummy)
            flat_dim = dummy.flatten(1).shape[1]

        self.proj = nn.Linear(flat_dim, embed_dim)
        self.final_ln = nn.LayerNorm(embed_dim) if final_ln else nn.Identity()
        self.embed_dim = embed_dim

    def forward(self, x):
        """Args: x (B, C, H, W).  Returns: (B, embed_dim)."""
        for stack in self.stacks:
            x = stack(x)
        x = F.relu(x)
        x = self.proj(x.flatten(1))
        return self.final_ln(x)


# ---------------------------------------------------------------------------
# World model
# ---------------------------------------------------------------------------

class PLDMWM(nn.Module):
    """World model: IMPALA CNN encoder + GRU predictor.

    Interface mirrors DinoWM so the BeamSearchSolver and eval_wm.py work
    without modification.

    The GRU operates as a one-step transition model:
        next_emb = GRU(act_emb, current_emb_as_hidden)

    ``history_size`` is set to 1 because the GRU only needs the single most
    recent embedding as its hidden state (past context is implicit in the
    embedding, not explicit context window).
    """

    def __init__(
        self,
        encoder: ImpalaEncoder,
        action_encoder: nn.Module,
        action_emb_dim: int,
        frameskip: int = 1,
    ):
        super().__init__()
        self.encoder = encoder
        self.action_encoder = action_encoder
        self.frameskip = frameskip
        # history_size=1: the solver passes a single embedding as context
        self.history_size = 1

        embed_dim = encoder.embed_dim
        self.gru = nn.GRU(input_size=action_emb_dim, hidden_size=embed_dim, num_layers=1, batch_first=False)
        self.out_ln = nn.LayerNorm(embed_dim)

    # ------------------------------------------------------------------
    # Interface required by BeamSearchSolver / eval_wm.py
    # ------------------------------------------------------------------

    def get_cost(self, info_dict, action_candidates):
        raise NotImplementedError("PLDMWM uses BeamSearchSolver directly; get_cost is not supported.")

    def action_trans(self, x):
        """Tile action by frameskip along the last dim (matches DinoWM)."""
        return x.repeat(1, 1, self.frameskip)

    def encode_pixels(self, pixels):
        """Encode pixels to embeddings.

        Args: pixels (B, T, C, H, W) or (B, C, H, W)
        Returns: (B, T, embed_dim)
        """
        if pixels.dim() == 4:
            pixels = pixels.unsqueeze(1)
        b, t = pixels.shape[:2]
        flat = rearrange(pixels, "b t c h w -> (b t) c h w")
        emb = self.encoder(flat)                                    # (B*T, D)
        return rearrange(emb, "(b t) d -> b t d", b=b, t=t)

    def predict(self, emb_hist, act_emb_hist):
        """Autoregressive GRU rollout.

        Uses only ``emb_hist[:, 0, :]`` as the initial hidden state; subsequent
        steps are fully autoregressive (no teacher forcing).

        Args:
            emb_hist:     (B, T, D)  — history of embeddings; first entry = h_0
            act_emb_hist: (B, T, A)  — action embeddings; one per rollout step
        Returns:
            (B, T, D)  — predicted next embedding at each step
        """
        h = emb_hist[:, 0, :].unsqueeze(0)          # (1, B, D)
        preds = []
        for i in range(act_emb_hist.shape[1]):
            act = act_emb_hist[:, i, :].unsqueeze(0) # (1, B, A)
            out, h = self.gru(act, h)                # out: (1, B, D)
            out = self.out_ln(out[0])                # (B, D)
            h = out.unsqueeze(0)                     # feed prediction as next hidden
            preds.append(out)
        return torch.stack(preds, dim=1)             # (B, T, D)

    # ------------------------------------------------------------------
    # Convenience: one teacher-forced step (used by the training forward)
    # ------------------------------------------------------------------

    def gru_step_teacher(self, act_emb, hidden_emb):
        """Single GRU step with externally supplied hidden state (teacher forcing).

        Args:
            act_emb:    (B, A)
            hidden_emb: (B, D)  — actual encoded frame used as hidden state
        Returns:
            (B, D) — predicted next embedding
        """
        h = hidden_emb.unsqueeze(0)         # (1, B, D)
        out, _ = self.gru(act_emb.unsqueeze(0), h)
        return self.out_ln(out[0])          # (B, D)
