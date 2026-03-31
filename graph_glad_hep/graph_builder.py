"""
graph_builder.py
----------------
Convert raw particle four-momenta from LHC collision events into
PyTorch Geometric Data objects for graph-level anomaly detection.

Also provides MockEventGenerator for offline / CI testing with no
credentials or real data required.
"""
from __future__ import annotations

import numpy as np
import torch
from torch_geometric.data import Data


# ---------------------------------------------------------------------------
# Core graph construction
# ---------------------------------------------------------------------------

def build_event_graph(
    pts: np.ndarray,
    etas: np.ndarray,
    phis: np.ndarray,
    energies: np.ndarray,
    k: int = 8,
    max_particles: int = 100,
) -> Data:
    """Convert a single collision event to a PyTorch Geometric Data object.

    Nodes are particles; edges are k-nearest-neighbours in (eta, phi) space
    plus one directed seed edge from every particle to the highest-pT particle.
    Node features: (pt_norm, eta, phi, e_norm, dR_to_jet_axis) -- shape (N, 5).
    Edge features: (d_eta, d_phi, dR) -- shape (E, 3).

    Args:
        pts:           Transverse momenta, shape (N,).
        etas:          Pseudorapidities, shape (N,).
        phis:          Azimuthal angles, shape (N,).
        energies:      Energies, shape (N,).
        k:             Number of nearest neighbours for edges.
        max_particles: Clip event to top-pT particles if N > max_particles.

    Returns:
        torch_geometric.data.Data with fields x, edge_index, edge_attr.
    """
    # Clip to top-pT particles
    if len(pts) > max_particles:
        idx = np.argsort(pts)[::-1][:max_particles]
        pts, etas, phis, energies = pts[idx], etas[idx], phis[idx], energies[idx]

    N = len(pts)
    sum_pt = pts.sum() + 1e-9
    sum_e = energies.sum() + 1e-9

    # Jet axis: pT-weighted centroid in (eta, phi)
    eta_axis = (pts * etas).sum() / sum_pt
    phi_axis = (pts * phis).sum() / sum_pt
    dR_axis = np.sqrt((etas - eta_axis) ** 2 + (phis - phi_axis) ** 2)

    x = torch.tensor(
        np.stack([pts / sum_pt, etas, phis, energies / sum_e, dR_axis], axis=1),
        dtype=torch.float,
    )

    # k-NN edges in (eta, phi)
    pos = torch.tensor(np.stack([etas, phis], axis=1), dtype=torch.float)
    edge_index = _knn_edges(pos, k=min(k, N - 1))

    # Seed edges: every particle -> highest-pT particle
    seed_idx = int(pts.argmax())
    non_seed = [i for i in range(N) if i != seed_idx]
    if non_seed:
        seed_src = torch.tensor(non_seed, dtype=torch.long)
        seed_dst = torch.full((len(non_seed),), seed_idx, dtype=torch.long)
        seed_edges = torch.stack([seed_src, seed_dst], dim=0)
        edge_index = torch.cat([edge_index, seed_edges], dim=1)

    # Edge features
    src, dst = edge_index
    d_eta = x[src, 1] - x[dst, 1]
    d_phi = x[src, 2] - x[dst, 2]
    d_R = torch.sqrt(d_eta ** 2 + d_phi ** 2)
    edge_attr = torch.stack([d_eta, d_phi, d_R], dim=1)

    return Data(x=x, edge_index=edge_index, edge_attr=edge_attr, num_nodes=N)


def _knn_edges(pos: torch.Tensor, k: int) -> torch.Tensor:
    """Build k-NN edge_index from 2-D positions without requiring torch-cluster."""
    N = pos.size(0)
    # Pairwise squared distances
    diff = pos.unsqueeze(0) - pos.unsqueeze(1)          # (N, N, 2)
    dist2 = (diff ** 2).sum(dim=2)                       # (N, N)
    dist2.fill_diagonal_(float("inf"))
    # k nearest for each node
    _, nn_idx = dist2.topk(k, dim=1, largest=False)     # (N, k)
    src = torch.arange(N).unsqueeze(1).expand(-1, k).reshape(-1)
    dst = nn_idx.reshape(-1)
    return torch.stack([src, dst], dim=0)


# ---------------------------------------------------------------------------
# Mock data generator (no credentials / real data needed)
# ---------------------------------------------------------------------------

class MockEventGenerator:
    """Generate synthetic LHC-like collision events for testing and CI.

    Events follow a simplified QCD jet model:
      - Number of particles ~ Poisson(lam=n_particles_mean)
      - pT ~ exponential decay from a seed pT
      - (eta, phi) ~ Gaussian spread around jet axis

    Usage::

        gen = MockEventGenerator(seed=42)
        data = gen.generate_dataset(n_events=1000, signal_fraction=0.1)
        # data is a list of (torch_geometric.data.Data, label) tuples
    """

    def __init__(
        self,
        n_particles_mean: int = 30,
        pt_scale: float = 10.0,
        eta_spread: float = 0.4,
        phi_spread: float = 0.4,
        seed: int = 0,
    ) -> None:
        self.n_particles_mean = n_particles_mean
        self.pt_scale = pt_scale
        self.eta_spread = eta_spread
        self.phi_spread = phi_spread
        self.rng = np.random.default_rng(seed)

    def generate_event(self, is_signal: bool = False) -> Data:
        """Generate one synthetic event graph.

        Signal events have an additional hard-scatter substructure
        (two sub-clusters offset in phi) mimicking a boosted two-prong decay.
        """
        N = max(3, self.rng.poisson(self.n_particles_mean))
        eta_axis = self.rng.uniform(-2.5, 2.5)
        phi_axis = self.rng.uniform(-np.pi, np.pi)

        if is_signal:
            # Two-prong: split particles between two cores
            split = N // 2
            etas = np.concatenate([
                self.rng.normal(eta_axis + 0.2, self.eta_spread / 2, split),
                self.rng.normal(eta_axis - 0.2, self.eta_spread / 2, N - split),
            ])
            phis = np.concatenate([
                self.rng.normal(phi_axis + 0.3, self.phi_spread / 2, split),
                self.rng.normal(phi_axis - 0.3, self.phi_spread / 2, N - split),
            ])
        else:
            etas = self.rng.normal(eta_axis, self.eta_spread, N)
            phis = self.rng.normal(phi_axis, self.phi_spread, N)

        pts = self.rng.exponential(self.pt_scale, N)
        energies = pts * np.cosh(etas)

        return build_event_graph(pts, etas, phis, energies)

    def generate_dataset(
        self,
        n_events: int = 1000,
        signal_fraction: float = 0.1,
    ) -> list[tuple[Data, int]]:
        """Generate a labelled dataset of event graphs.

        Returns:
            List of (Data, label) where label=0 is background, label=1 is signal.
        """
        n_signal = int(n_events * signal_fraction)
        n_background = n_events - n_signal
        dataset = (
            [(self.generate_event(is_signal=False), 0) for _ in range(n_background)]
            + [(self.generate_event(is_signal=True), 1) for _ in range(n_signal)]
        )
        self.rng.shuffle(dataset)
        return dataset
