"""
encoder.py
----------
Dual GCN encoder for graph-level anomaly detection.

Architecture follows GLADC (Luo et al., 2022, Scientific Reports 12:19867)
with HEP-specific adaptations:
  - Combined mean+max graph readout (more stable on variable-size jet graphs)
  - Three-layer projection head matching LHCO baseline dimensionality
  - Gaussian weight perturbation for augmentation-free contrastive learning
"""
from __future__ import annotations

import torch
import torch.nn as nn
from torch_geometric.nn import GCNConv, global_mean_pool, global_max_pool


class GraphEncoder(nn.Module):
    """Two-layer GCN encoder.

    Produces node-level embeddings Z_node (N x out_dim) and graph-level
    embeddings Z_G (B x 2*out_dim) via concatenated mean+max pooling.

    Args:
        in_dim:  Input node feature dimension.
        hidden:  Hidden layer width.
        out_dim: Output embedding dimension.
    """

    def __init__(self, in_dim: int, hidden: int = 256, out_dim: int = 128) -> None:
        super().__init__()
        self.conv1 = GCNConv(in_dim, hidden)
        self.conv2 = GCNConv(hidden, out_dim)
        self.act = nn.ReLU()
        self.in_dim = in_dim
        self.hidden = hidden
        self.out_dim = out_dim

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        batch: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Forward pass.

        Args:
            x:          Node features (N_total x in_dim).
            edge_index: Edge indices (2 x E).
            batch:      Batch assignment vector (N_total,).

        Returns:
            (z_node, z_g): Node embeddings (N x out_dim),
                           graph embeddings (B x 2*out_dim).
        """
        z = self.act(self.conv1(x, edge_index))
        z = self.conv2(z, edge_index)
        z_g = torch.cat(
            [global_mean_pool(z, batch), global_max_pool(z, batch)], dim=1
        )
        return z, z_g


class GCNDecoder(nn.Module):
    """Node attribute decoder using a single GCN layer."""

    def __init__(self, in_dim: int, out_dim: int) -> None:
        super().__init__()
        self.conv = GCNConv(in_dim, out_dim)

    def forward(
        self,
        z: torch.Tensor,
        edge_index: torch.Tensor,
    ) -> torch.Tensor:
        return self.conv(z, edge_index)


class DualEncoder(nn.Module):
    """GLADC dual encoder: clean branch + weight-perturbed branch + projection head.

    The perturbed encoder is obtained by adding scaled Gaussian noise to
    all encoder weight matrices (SimGRACE-style, no data augmentation).

    Args:
        in_dim:  Input node feature dimension (must match graph_builder output = 5).
        hidden:  GCN hidden width.
        out_dim: GCN output embedding dimension.
        eta:     Perturbation strength coefficient.
    """

    def __init__(
        self,
        in_dim: int = 5,
        hidden: int = 256,
        out_dim: int = 128,
        eta: float = 1.0,
    ) -> None:
        super().__init__()
        self.encoder = GraphEncoder(in_dim, hidden, out_dim)
        self.decoder_attr = GCNDecoder(out_dim, in_dim)
        self.eta = eta
        self.out_dim = out_dim

        graph_emb_dim = 2 * out_dim          # concat mean + max
        self.proj = nn.Sequential(
            nn.Linear(graph_emb_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
        )

    def _build_perturbed_encoder(self) -> GraphEncoder:
        """Instantiate an encoder with Gaussian-perturbed weights (no grad)."""
        enc_hat = GraphEncoder(
            self.encoder.in_dim,
            self.encoder.hidden,
            self.encoder.out_dim,
        ).to(next(self.parameters()).device)
        enc_hat.load_state_dict(self.encoder.state_dict())
        with torch.no_grad():
            for p_orig, p_hat in zip(
                self.encoder.parameters(), enc_hat.parameters()
            ):
                sigma = p_orig.data.std().clamp(min=1e-6)
                noise = torch.randn_like(p_orig) * sigma
                p_hat.data.add_(self.eta * noise)
        return enc_hat

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        batch: torch.Tensor,
    ) -> tuple:
        """Forward pass through both branches.

        Returns:
            Tuple of:
              z_node   -- clean node embeddings (N x out_dim)
              z_g      -- clean graph embeddings (B x 2*out_dim)
              zhat_node -- perturbed node embeddings
              zhat_g   -- perturbed graph embeddings
              h_g      -- projected clean graph embeddings (B x 64)
              hhat_g   -- projected perturbed graph embeddings (B x 64)
              x_hat    -- reconstructed node attributes (N x in_dim)
        """
        z_node, z_g = self.encoder(x, edge_index, batch)
        x_hat = self.decoder_attr(z_node, edge_index)

        enc_hat = self._build_perturbed_encoder()
        zhat_node, zhat_g = enc_hat(x, edge_index, batch)

        h_g = self.proj(z_g)
        hhat_g = self.proj(zhat_g)

        return z_node, z_g, zhat_node, zhat_g, h_g, hhat_g, x_hat
