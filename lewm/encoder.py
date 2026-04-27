import torch.nn as nn
from timm.models.vision_transformer import VisionTransformer


class Encoder(nn.Module):
    """
    ViT-Tiny encoder + 1-layer MLP projector.

    Architecture (paper Section 3.1):
      - ViT-Tiny: patch_size=14, depth=12, heads=3, embed_dim=192  (~5M params)
      - Projector: Linear → BatchNorm  (not LayerNorm!)

    WHY BatchNorm in projector:
      ViT's final layer applies LayerNorm, which fixes per-sample variance to 1.
      SIGReg needs to measure diversity *across the batch*, so we need a
      normalization that preserves batch-level statistics. BatchNorm does this;
      LayerNorm would undo it.
    """

    def __init__(self, embed_dim: int = 192):
        super().__init__()
        self.vit = VisionTransformer(
            img_size=224,
            patch_size=14,
            in_chans=3,
            num_classes=0,       # remove classification head, return features
            embed_dim=embed_dim,
            depth=12,
            num_heads=3,
            mlp_ratio=4.0,
            qkv_bias=True,
            global_pool="token", # return CLS token, shape (B, embed_dim)
        )
        self.projector = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.BatchNorm1d(embed_dim),
        )

    def forward(self, x):
        # x: (B, 3, 224, 224)
        cls = self.vit(x)           # (B, embed_dim)
        return self.projector(cls)  # (B, embed_dim)
