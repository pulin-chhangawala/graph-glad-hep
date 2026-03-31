"""
model.py
--------
Scikit-learn compatible GraphGLADC estimator.

Wraps the DualEncoder training loop behind a fit() / score_samples() API
so it can be dropped into any ML4SCI evaluation pipeline without modification.
"""
from __future__ import annotations

import numpy as np
import torch
from torch_geometric.data import Data, DataLoader

from .encoder import DualEncoder
from .losses import total_loss, anomaly_scores


class GraphGLADC:
    """Graph-level anomaly detector using GLADC with contrastive learning.

    Parameters
    ----------
    in_dim : int
        Node feature dimensionality (default 5, matching graph_builder output).
    hidden : int
        GCN hidden layer width.
    out_dim : int
        GCN output embedding dimension.
    eta : float
        Perturbation strength for the perturbed encoder branch.
    tau : float
        Temperature for the InfoNCE contrastive loss.
    lambda_disco : float
        DisCo regularisation strength (0 = disabled).
    lr : float
        Adam learning rate.
    epochs : int
        Number of training epochs.
    batch_size : int
        Training mini-batch size (number of graphs per batch).
    device : str
        Torch device string ('cpu' or 'cuda').
    """

    def __init__(
        self,
        in_dim: int = 5,
        hidden: int = 256,
        out_dim: int = 128,
        eta: float = 1.0,
        tau: float = 0.2,
        lambda_disco: float = 0.0,
        lr: float = 1e-4,
        epochs: int = 50,
        batch_size: int = 128,
        device: str = "cpu",
    ) -> None:
        self.in_dim = in_dim
        self.hidden = hidden
        self.out_dim = out_dim
        self.eta = eta
        self.tau = tau
        self.lambda_disco = lambda_disco
        self.lr = lr
        self.epochs = epochs
        self.batch_size = batch_size
        self.device = torch.device(device)
        self.model_: DualEncoder | None = None

    def fit(self, graphs: list[Data], verbose: bool = True) -> "GraphGLADC":
        """Train on a list of background (normal) event graphs.

        Args:
            graphs:  List of torch_geometric.data.Data objects (background only).
            verbose: Print epoch loss if True.

        Returns:
            self (for sklearn-style chaining).
        """
        self.model_ = DualEncoder(
            in_dim=self.in_dim,
            hidden=self.hidden,
            out_dim=self.out_dim,
            eta=self.eta,
        ).to(self.device)

        optimiser = torch.optim.Adam(self.model_.parameters(), lr=self.lr)
        loader = DataLoader(graphs, batch_size=self.batch_size, shuffle=True)

        self.model_.train()
        for epoch in range(1, self.epochs + 1):
            epoch_loss = 0.0
            for batch in loader:
                batch = batch.to(self.device)
                optimiser.zero_grad()

                z_node, z_g, zhat_node, zhat_g, h_g, hhat_g, x_hat = self.model_(
                    batch.x, batch.edge_index, batch.batch
                )

                # Re-encode the reconstructed graph to get primed embeddings
                import torch_geometric.utils as pyg_utils
                # Use reconstructed attributes x_hat as proxy for fake graph
                zp_node, zp_g = self.model_.encoder(x_hat.detach(), batch.edge_index, batch.batch)

                loss = total_loss(
                    z_node=z_node, x=batch.x, x_hat=x_hat,
                    edge_index=batch.edge_index,
                    h=h_g, h_hat=hhat_g,
                    z_g=z_g, zp_g=zp_g, zp_node=zp_node,
                    batch=batch.batch, tau=self.tau,
                )
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model_.parameters(), 1.0)
                optimiser.step()
                epoch_loss += loss.item()

            if verbose and epoch % 10 == 0:
                print(f"Epoch {epoch:3d}/{self.epochs}  loss={epoch_loss/len(loader):.4f}")

        return self

    def score_samples(self, graphs: list[Data]) -> np.ndarray:
        """Compute anomaly score for each graph (higher = more anomalous).

        Args:
            graphs: List of torch_geometric.data.Data objects.

        Returns:
            Numpy array of shape (N,) with per-graph anomaly scores.
        """
        if self.model_ is None:
            raise RuntimeError("Call fit() before score_samples().")

        self.model_.eval()
        loader = DataLoader(graphs, batch_size=self.batch_size, shuffle=False)
        all_scores = []

        with torch.no_grad():
            for batch in loader:
                batch = batch.to(self.device)
                z_node, z_g, _, _, _, _, x_hat = self.model_(
                    batch.x, batch.edge_index, batch.batch
                )
                zp_node, zp_g = self.model_.encoder(
                    x_hat, batch.edge_index, batch.batch
                )
                scores = anomaly_scores(z_node, zp_node, z_g, zp_g, batch.batch)
                all_scores.append(scores.cpu().numpy())

        return np.concatenate(all_scores)
