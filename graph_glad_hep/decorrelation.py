"""
decorrelation.py
----------------
Distance Correlation (DisCo) regulariser for mass decorrelation.

Penalises statistical dependence between the anomaly score S(G) and the
dijet invariant mass m_jj, preventing the model from learning kinematic
artefacts instead of genuine structural anomalies.

Reference:
    Pfau et al., "DisCo Fever: Robust Networks through Distance Correlation",
    arXiv:2001.05566 (2020).
"""
from __future__ import annotations

import torch


def dist_corr(X: torch.Tensor, Y: torch.Tensor) -> torch.Tensor:
    """Differentiable distance correlation between anomaly score and mass.

    Distance correlation is zero if and only if X and Y are statistically
    independent. It is differentiable with respect to X, enabling gradient-
    based optimisation to push dCor -> 0.

    Args:
        X: Anomaly scores, shape (N,). Should be detached from the graph
           only for logging; keep gradients for training.
        Y: Dijet invariant masses (or other kinematic variable), shape (N,).

    Returns:
        Scalar distance correlation in [0, 1].
    """
    n = X.size(0)
    if n < 2:
        return torch.tensor(0.0, device=X.device, requires_grad=True)

    a = torch.cdist(X.unsqueeze(1).float(), X.unsqueeze(1).float(), p=1)
    b = torch.cdist(Y.unsqueeze(1).float(), Y.unsqueeze(1).float(), p=1)

    # Double-centre each distance matrix
    A = _double_centre(a)
    B = _double_centre(b)

    dcov2_XY = (A * B).sum() / (n * n)
    dcov2_XX = (A * A).sum() / (n * n)
    dcov2_YY = (B * B).sum() / (n * n)

    denom = (dcov2_XX.clamp(min=0).sqrt() * dcov2_YY.clamp(min=0).sqrt()).clamp(min=1e-9)
    dcor = dcov2_XY / denom
    return dcor.clamp(min=0.0)


def _double_centre(M: torch.Tensor) -> torch.Tensor:
    """Apply double-centring to a pairwise distance matrix."""
    row_mean = M.mean(dim=1, keepdim=True)
    col_mean = M.mean(dim=0, keepdim=True)
    grand_mean = M.mean()
    return M - row_mean - col_mean + grand_mean
