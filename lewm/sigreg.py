import torch
import torch.nn.functional as F


def _epps_pulley(h: torch.Tensor, nodes: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
    """
    Vectorized Epps-Pulley normality test across M projections.

    h:       (N, M)  — N samples projected onto M directions
    nodes:   (K,)    — quadrature nodes in [0.2, 4]
    weights: (K,)    — trapezoid weights * Gaussian weighting function

    Returns: (M,)    — one test statistic per projection direction

    The test measures how far the empirical characteristic function (ECF)
    of h deviates from the standard Gaussian CF exp(-t²/2).
    Cramer-Wold theorem: matching all 1D marginals ⟺ matching the full
    joint distribution, so averaging over M random projections is enough.
    """
    # t_h[k, n, m] = nodes[k] * h[n, m]
    t_h = nodes.view(-1, 1, 1) * h.unsqueeze(0)  # (K, N, M)

    ecf_real = t_h.cos().mean(1)  # (K, M)  Re[ECF(t)]
    ecf_imag = t_h.sin().mean(1)  # (K, M)  Im[ECF(t)]

    # Standard Gaussian CF: φ₀(t) = exp(-t²/2), purely real
    gauss_cf = (-0.5 * nodes ** 2).exp()  # (K,)

    # |ECF(t) - φ₀(t)|²
    integrand = (ecf_real - gauss_cf.view(-1, 1)) ** 2 + ecf_imag ** 2  # (K, M)

    # Integrate over t with trapezoid quadrature (weights already include w(t))
    return (integrand * weights.view(-1, 1)).sum(0)  # (M,)


def sigreg(Z: torch.Tensor, M: int = 1024, n_nodes: int = 32) -> torch.Tensor:
    """
    Sketched-Isotropic-Gaussian Regularizer (SIGReg).

    Z:       (T, B, D)  — embeddings at T timesteps, batch B, dim D
             Applied per-timestep then averaged, matching Algorithm 1:
             mean(SIGReg(emb.transpose(0, 1)))

    M:       number of random projection directions (paper default: 1024)
    n_nodes: quadrature points for the integral (paper: trapezoid in [0.2, 4])

    Returns: scalar loss
    """
    T, B, D = Z.shape

    # --- Quadrature setup (done once, shared across timesteps) ---
    nodes = torch.linspace(0.2, 4.0, n_nodes, device=Z.device)   # (K,)

    # Trapezoid weights: h for interior, h/2 for endpoints
    step = (4.0 - 0.2) / (n_nodes - 1)
    trap_w = torch.full((n_nodes,), step, device=Z.device)
    trap_w[0] *= 0.5
    trap_w[-1] *= 0.5

    # Weighting function w(t) = exp(-t²/2), absorbed into quadrature weights
    weights = trap_w * (-0.5 * nodes ** 2).exp()  # (K,)

    # --- Random unit-norm projection directions u ∈ S^{D-1} ---
    # Re-sampled each forward pass: stochastic projections act as data augmentation
    # and prevent the model from memorizing specific directions.
    u = F.normalize(torch.randn(D, M, device=Z.device), dim=0)  # (D, M)

    # --- Per-timestep SIGReg, then average ---
    total = Z.new_zeros(())
    for t in range(T):
        h = Z[t] @ u  # (B, M)
        total = total + _epps_pulley(h, nodes, weights).mean()

    return total / T
