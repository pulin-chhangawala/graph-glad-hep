"""
losses.py
---------
Loss functions for the GLADC framework applied to HEP data.

Three losses are combined into the total training objective:
  L1 -- GCN autoencoder reconstruction (structure + attributes)
  L2 -- SimGRACE-style InfoNCE contrastive loss on graph embeddings
  L3 -- Representation error (anomaly score at training time)
"""
from __future__ import annotations

import torch
import torch.nn.functional as F
from torch_scatter import scatter


def reconstruction_loss(
    z_node: torch.Tensor,
    x: torch.Tensor,
    x_hat: torch.Tensor,
    edge_index: torch.Tensor,
    batch: torch.Tensor,
) -> torch.Tensor:
    """GCN autoencoder reconstruction loss (L1).

    Combines:
      - Adjacency reconstruction via inner product decoder (BCE)
      - Node attribute reconstruction (MSE)

    Args:
        z_node:     Node embeddings from encoder (N x D).
        x:          Original node features (N x in_dim).
        x_hat:      Reconstructed node features from decoder (N x in_dim).
        edge_index: Edge connectivity (2 x E).
        batch:      Batch assignment (N,).

    Returns:
        Scalar loss value.
    """
    # Attribute reconstruction (MSE)
    L_attr = F.mse_loss(x_hat, x)

    # Structural reconstruction per graph (inner product over edges only)
    src, dst = edge_index
    dot = (z_node[src] * z_node[dst]).sum(dim=1)
    adj_pred = torch.sigmoid(dot)
    adj_target = torch.ones_like(adj_pred)   # all edges exist by construction
    L_struct = F.binary_cross_entropy(adj_pred, adj_target)

    return L_attr + L_struct


def contrastive_loss(
    h: torch.Tensor,
    h_hat: torch.Tensor,
    temperature: float = 0.2,
) -> torch.Tensor:
    """SimGRACE-style InfoNCE contrastive loss (L2).

    Positive pairs: (h_i, hhat_i) -- same graph, clean vs perturbed encoder.
    Negative pairs: all other hhat_j in the batch (j != i).

    Args:
        h:           Projected clean graph embeddings (B x D).
        h_hat:       Projected perturbed graph embeddings (B x D).
        temperature: Softmax temperature tau.

    Returns:
        Scalar contrastive loss.
    """
    h = F.normalize(h, dim=1)
    h_hat = F.normalize(h_hat, dim=1)
    sim = (h @ h_hat.T) / temperature          # (B x B)
    labels = torch.arange(h.size(0), device=h.device)
    return F.cross_entropy(sim, labels)


def representation_error(
    z_node: torch.Tensor,
    zp_node: torch.Tensor,
    z_g: torch.Tensor,
    zp_g: torch.Tensor,
    batch: torch.Tensor,
) -> torch.Tensor:
    """Representation error between input-graph and reconstructed-graph codes (L3).

    This is also the anomaly score (equation 11 in GLADC paper).

    Args:
        z_node:  Node embeddings from real input graph (N x D).
        zp_node: Node embeddings from reconstructed graph (N x D).
        z_g:     Graph embeddings from real input graph (B x 2D).
        zp_g:    Graph embeddings from reconstructed graph (B x 2D).
        batch:   Batch assignment vector (N,).

    Returns:
        Scalar mean loss across the batch.
    """
    node_err = ((z_node - zp_node) ** 2).sum(dim=1)         # (N,)
    graph_node_err = scatter(node_err, batch, reduce="mean")  # (B,)
    graph_err = ((z_g - zp_g) ** 2).sum(dim=1)              # (B,)
    return (graph_node_err + graph_err).mean()


def anomaly_scores(
    z_node: torch.Tensor,
    zp_node: torch.Tensor,
    z_g: torch.Tensor,
    zp_g: torch.Tensor,
    batch: torch.Tensor,
) -> torch.Tensor:
    """Per-graph anomaly scores (not reduced -- for inference).

    Returns:
        Score tensor of shape (B,).
    """
    node_err = ((z_node - zp_node) ** 2).sum(dim=1)
    graph_node_err = scatter(node_err, batch, reduce="mean")
    graph_err = ((z_g - zp_g) ** 2).sum(dim=1)
    return graph_node_err + graph_err


def total_loss(
    z_node: torch.Tensor,
    x: torch.Tensor,
    x_hat: torch.Tensor,
    edge_index: torch.Tensor,
    h: torch.Tensor,
    h_hat: torch.Tensor,
    z_g: torch.Tensor,
    zp_g: torch.Tensor,
    zp_node: torch.Tensor,
    batch: torch.Tensor,
    tau: float = 0.2,
    lambda_disco: float = 0.0,
    scores: torch.Tensor | None = None,
    masses: torch.Tensor | None = None,
) -> torch.Tensor:
    """Combined GLADC training loss with optional DisCo regularisation.

    L_total = L1 + L2 + L3 [+ lambda * dCor^2(score, mass)]

    Args:
        lambda_disco: DisCo regularisation strength (0 = disabled).
        scores:       Per-graph anomaly scores, required if lambda_disco > 0.
        masses:       Dijet invariant masses, required if lambda_disco > 0.
    """
    L1 = reconstruction_loss(z_node, x, x_hat, edge_index, batch)
    L2 = contrastive_loss(h, h_hat, tau)
    L3 = representation_error(z_node, zp_node, z_g, zp_g, batch)
    loss = L1 + L2 + L3

    if lambda_disco > 0 and scores is not None and masses is not None:
        from .decorrelation import dist_corr
        loss = loss + lambda_disco * dist_corr(scores, masses) ** 2

    return loss
