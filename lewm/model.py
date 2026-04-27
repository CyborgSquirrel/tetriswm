import torch
import torch.nn as nn
import torch.nn.functional as F

from .encoder import Encoder
from .predictor import Predictor
from .sigreg import sigreg


class LeWM(nn.Module):
    """
    LeWorldModel: end-to-end JEPA world model.

    Training objective (paper Eq. 3):
        L = L_pred + λ · SIGReg(Z)

    L_pred: MSE between predicted and actual next embedding (teacher-forcing).
            Forces the predictor to learn environment dynamics.

    SIGReg: Pushes the embedding distribution toward an isotropic Gaussian.
            Prevents representation collapse without stop-gradient or EMA.

    λ = 0.1 works well across a wide range [0.01, 0.2] (paper Fig. 16).
    """

    def __init__(
        self,
        n_actions: int,
        embed_dim: int = 192,
        lambda_reg: float = 0.1,
    ):
        super().__init__()
        self.encoder = Encoder(embed_dim)
        self.predictor = Predictor(embed_dim=embed_dim, n_actions=n_actions)
        self.lambda_reg = lambda_reg

    def forward(self, obs, actions):
        """
        obs:     (B, T, 3, 224, 224)  — pixel frames
        actions: (B, T)               — discrete actions (integers)

        Returns: dict with 'loss', 'pred_loss', 'sigreg_loss'
        """
        B, T, C, H, W = obs.shape

        # Encode all frames in one batch pass: (B*T, C, H, W) → (B*T, D)
        emb = self.encoder(obs.view(B * T, C, H, W)).view(B, T, -1)  # (B, T, D)

        # Predict next embeddings (teacher-forcing: feed ground-truth embeddings)
        # Output at position t is prediction for position t+1
        next_emb = self.predictor(emb, actions)  # (B, T, D)

        # --- Prediction loss ---
        # next_emb[:, t] predicts emb[:, t+1], so compare [:-1] with [1:]
        # No detach: gradients flow through both encoder and predictor jointly.
        pred_loss = F.mse_loss(next_emb[:, :-1], emb[:, 1:])

        # --- SIGReg loss ---
        # emb.transpose(0,1): (B,T,D) → (T,B,D) = (N,B,D) format for sigreg()
        sigreg_loss = sigreg(emb.transpose(0, 1))

        loss = pred_loss + self.lambda_reg * sigreg_loss

        return {
            "loss": loss,
            "pred_loss": pred_loss.detach(),
            "sigreg_loss": sigreg_loss.detach(),
        }

    @torch.no_grad()
    def encode(self, obs):
        """Encode a single frame: (B, 3, 224, 224) → (B, D)"""
        return self.encoder(obs)

    @torch.no_grad()
    def step(self, z, action):
        """
        Single-step latent rollout for planning.
        z:      (B, D)  current latent state
        action: (B,)    discrete action
        Returns: (B, D) predicted next latent state
        """
        # Predictor expects (B, T, D) and (B, T); use T=1
        z_seq = z.unsqueeze(1)                     # (B, 1, D)
        a_seq = action.unsqueeze(1)                # (B, 1)
        return self.predictor(z_seq, a_seq)[:, 0]  # (B, D)
