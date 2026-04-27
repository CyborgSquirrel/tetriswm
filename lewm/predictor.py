import torch
import torch.nn as nn


class AdaLNBlock(nn.Module):
    """
    Transformer block with Adaptive Layer Normalization (AdaLN).

    Standard LN: y = γ · norm(x) + β,  γ=1, β=0 (fixed)
    AdaLN:       γ and β are predicted from the action embedding,
                 so the action can shift and scale every layer's features.

    Inspired by DiT (Peebles & Xie, 2023). The modulation MLP is
    zero-initialized so at t=0 the block is an identity — training is
    stable from the start and action influence grows gradually.
    """

    def __init__(self, d_model: int, n_heads: int, d_action: int, dropout: float = 0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.ff = nn.Sequential(
            nn.Linear(d_model, 4 * d_model),
            nn.GELU(),
            nn.Linear(4 * d_model, d_model),
            nn.Dropout(dropout),
        )
        # Predicts (shift1, scale1, shift2, scale2) from action embedding.
        # Zero-init: at init, all modulation = 0, block = identity.
        self.adaLN = nn.Linear(d_action, 4 * d_model)
        nn.init.zeros_(self.adaLN.weight)
        nn.init.zeros_(self.adaLN.bias)

    def forward(self, x, action_emb, causal_mask=None):
        # x:          (B, T, d_model)
        # action_emb: (B, T, d_action)
        shift1, scale1, shift2, scale2 = self.adaLN(action_emb).chunk(4, dim=-1)

        # Attention sub-block
        x_norm = self.norm1(x) * (1 + scale1) + shift1
        attn_out, _ = self.attn(x_norm, x_norm, x_norm, attn_mask=causal_mask,
                                is_causal=True)
        x = x + attn_out

        # Feed-forward sub-block
        x_norm = self.norm2(x) * (1 + scale2) + shift2
        x = x + self.ff(x_norm)
        return x


class Predictor(nn.Module):
    """
    Action-conditioned latent dynamics model.

    Takes a sequence of embeddings z_{1:T} and discrete actions a_{1:T},
    predicts next embeddings ẑ_{2:T+1} autoregressively with causal masking.

    Architecture (paper Section 3.1):
      - Discrete action → embedding (Tetris adaptation: paper used continuous)
      - 6 AdaLN Transformer layers, 16 heads, 10% dropout  (~10M params)
      - Same projector design as encoder (Linear + BatchNorm)
    """

    def __init__(
        self,
        embed_dim: int = 192,
        n_actions: int = 5,
        d_action: int = 64,
        n_layers: int = 6,
        n_heads: int = 16,
        dropout: float = 0.1,
        max_seq_len: int = 64,
    ):
        super().__init__()
        # Embed discrete Tetris actions into a continuous vector.
        # The paper uses continuous actions directly; for discrete we learn embeddings.
        self.action_emb = nn.Embedding(n_actions, d_action)

        # Learned positional embeddings (causal transformer needs position info)
        self.pos_emb = nn.Parameter(torch.zeros(1, max_seq_len, embed_dim))

        self.blocks = nn.ModuleList([
            AdaLNBlock(embed_dim, n_heads, d_action, dropout)
            for _ in range(n_layers)
        ])

        # Same projector structure as encoder for symmetry
        self.projector = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.BatchNorm1d(embed_dim),
        )

    def forward(self, z, actions):
        # z:       (B, T, embed_dim)
        # actions: (B, T)  — integers in [0, n_actions)
        B, T, D = z.shape

        a = self.action_emb(actions)         # (B, T, d_action)
        x = z + self.pos_emb[:, :T]          # add positional info

        # Upper-triangular mask: position i cannot attend to j > i
        causal_mask = torch.triu(
            torch.full((T, T), float("-inf"), device=z.device), diagonal=1
        )

        for block in self.blocks:
            x = block(x, a, causal_mask)

        # Apply projector timestep-wise: flatten (B,T,D) → (B*T, D) for BatchNorm
        out = self.projector(x.reshape(B * T, D)).reshape(B, T, D)
        return out
