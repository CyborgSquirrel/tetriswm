"""DinoWM model class — kept in its own module so torch.load can unpickle it
regardless of which script is __main__."""

import sys
import glob
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "dinowm" / "models"))

import torch

for _d in glob.glob(str(Path(torch.hub.get_dir()) / "facebookresearch_dinov2*")):
    sys.path.insert(0, _d)
import torch.nn as nn
from einops import rearrange

from dino import DinoV2Encoder  # noqa: F401 — needed for unpickling checkpoints


class DinoWM(nn.Module):
    """Wraps frozen DINO encoder + ViTPredictor + action encoder."""

    def __init__(self, encoder, predictor, action_encoder,
                 history_size, dino_dim, num_patches, concat_dim, frameskip):
        super().__init__()
        self.encoder = encoder
        self.predictor = predictor
        self.action_encoder = action_encoder
        self.history_size = history_size
        self.dino_dim = dino_dim
        self.num_patches = num_patches
        self.concat_dim = concat_dim
        self.frameskip = frameskip

    def action_trans(self, x):
        """Tile action by frameskip along the last dim."""
        return x.repeat(1, 1, self.frameskip) if self.frameskip > 1 else x

    def encode_pixels(self, pixels):
        """Encode pixels to flattened DINO patch tokens → (B, T, P*D_patch)."""
        if pixels.dim() == 4:
            pixels = pixels.unsqueeze(1)
        b, t = pixels.shape[:2]
        pixels_flat = rearrange(pixels, "b t c h w -> (b t) c h w")
        with torch.no_grad():
            patch_embs_flat = self.encoder(pixels_flat)  # (B*T, P, D)
        return rearrange(patch_embs_flat, "(b t) p d -> b t (p d)", b=b, t=t)

    def predict(self, emb_hist, act_emb_hist):
        """Predict next visual embeddings.

        Args:
            emb_hist:     (B, T, P*D_patch) flattened visual embeddings
            act_emb_hist: (B, T, A_emb) action embeddings
        Returns:
            (B, T, P*D_patch)
        """
        b, t, _ = emb_hist.shape
        p, d = self.num_patches, self.dino_dim

        patch_embs = rearrange(emb_hist, "b t (p d) -> b t p d", p=p, d=d)

        if self.concat_dim == 1:
            act_repeated = act_emb_hist.unsqueeze(2).expand(-1, -1, p, -1)
            z = torch.cat([patch_embs, act_repeated], dim=3)
            tokens_per_frame = p
        else:
            z = torch.cat([patch_embs, act_emb_hist.unsqueeze(2)], dim=2)
            tokens_per_frame = p + 1

        z_flat = rearrange(z, "b t p d -> b (t p) d")
        z_pred_flat = self.predictor(z_flat)
        z_pred = rearrange(z_pred_flat, "b (t p) d -> b t p d", t=t, p=tokens_per_frame)

        if self.concat_dim == 1:
            visual_pred = z_pred[:, :, :, :d]
        else:
            visual_pred = z_pred[:, :, :-1, :]

        return rearrange(visual_pred, "b t p d -> b t (p d)")
