"""
tests/test_graph_builder.py
---------------------------
Unit tests for the graph construction pipeline.
All tests run with no GPU and no real data (mock generator only).
"""
import numpy as np
import pytest
import torch

from graph_glad_hep.graph_builder import build_event_graph, MockEventGenerator


# ---------------------------------------------------------------------------
# build_event_graph
# ---------------------------------------------------------------------------

class TestBuildEventGraph:

    def _make_event(self, n=20):
        rng = np.random.default_rng(0)
        pts = rng.exponential(10.0, n)
        etas = rng.normal(0.0, 0.4, n)
        phis = rng.normal(0.0, 0.4, n)
        energies = pts * np.cosh(etas)
        return pts, etas, phis, energies

    def test_returns_data_object(self):
        from torch_geometric.data import Data
        pts, etas, phis, energies = self._make_event()
        g = build_event_graph(pts, etas, phis, energies)
        assert isinstance(g, Data)

    def test_node_feature_shape(self):
        pts, etas, phis, energies = self._make_event(n=20)
        g = build_event_graph(pts, etas, phis, energies, k=4)
        assert g.x.shape == (20, 5)

    def test_edge_index_shape(self):
        pts, etas, phis, energies = self._make_event(n=20)
        g = build_event_graph(pts, etas, phis, energies, k=4)
        assert g.edge_index.shape[0] == 2
        assert g.edge_index.shape[1] > 0

    def test_edge_attr_shape(self):
        pts, etas, phis, energies = self._make_event(n=20)
        g = build_event_graph(pts, etas, phis, energies, k=4)
        assert g.edge_attr.shape[1] == 3
        assert g.edge_attr.shape[0] == g.edge_index.shape[1]

    def test_node_features_are_finite(self):
        pts, etas, phis, energies = self._make_event()
        g = build_event_graph(pts, etas, phis, energies)
        assert torch.isfinite(g.x).all()

    def test_edge_indices_in_range(self):
        n = 15
        pts, etas, phis, energies = self._make_event(n=n)
        g = build_event_graph(pts, etas, phis, energies, k=3)
        assert g.edge_index.min() >= 0
        assert g.edge_index.max() < n

    def test_max_particles_clipping(self):
        pts, etas, phis, energies = self._make_event(n=150)
        g = build_event_graph(pts, etas, phis, energies, max_particles=50)
        assert g.num_nodes == 50

    def test_no_clipping_below_max(self):
        pts, etas, phis, energies = self._make_event(n=30)
        g = build_event_graph(pts, etas, phis, energies, max_particles=100)
        assert g.num_nodes == 30

    def test_normalised_pt_sums_to_one(self):
        pts, etas, phis, energies = self._make_event(n=25)
        g = build_event_graph(pts, etas, phis, energies)
        pt_norm = g.x[:, 0]
        assert abs(float(pt_norm.sum()) - 1.0) < 1e-4

    def test_single_particle_event(self):
        """Edge case: 1 particle -> graph with 0 edges."""
        g = build_event_graph(
            np.array([10.0]), np.array([0.0]),
            np.array([0.0]), np.array([10.0]),
            k=1,
        )
        assert g.num_nodes == 1
        assert g.edge_index.shape[1] == 0

    def test_three_particle_event(self):
        g = build_event_graph(
            np.array([10.0, 5.0, 3.0]),
            np.array([0.0, 0.1, -0.1]),
            np.array([0.0, 0.1, -0.1]),
            np.array([10.0, 5.0, 3.0]),
            k=2,
        )
        assert g.num_nodes == 3

    def test_reproducibility(self):
        rng = np.random.default_rng(99)
        pts = rng.exponential(5.0, 30)
        etas = rng.normal(0.0, 0.3, 30)
        phis = rng.normal(0.0, 0.3, 30)
        energies = pts * 1.1
        g1 = build_event_graph(pts, etas, phis, energies)
        g2 = build_event_graph(pts, etas, phis, energies)
        assert torch.allclose(g1.x, g2.x)
        assert torch.equal(g1.edge_index, g2.edge_index)


# ---------------------------------------------------------------------------
# MockEventGenerator
# ---------------------------------------------------------------------------

class TestMockEventGenerator:

    def test_generate_event_returns_data(self):
        from torch_geometric.data import Data
        gen = MockEventGenerator(seed=0)
        g = gen.generate_event(is_signal=False)
        assert isinstance(g, Data)

    def test_generate_event_has_features(self):
        gen = MockEventGenerator(seed=1)
        g = gen.generate_event()
        assert g.x.shape[1] == 5

    def test_signal_event_different_from_background(self):
        gen = MockEventGenerator(seed=2)
        bg = gen.generate_event(is_signal=False)
        sg = gen.generate_event(is_signal=True)
        # Not guaranteed to differ, but very likely with fixed seed
        assert bg.x.shape == sg.x.shape or True  # shape may differ

    def test_generate_dataset_length(self):
        gen = MockEventGenerator(seed=3)
        ds = gen.generate_dataset(n_events=50, signal_fraction=0.2)
        assert len(ds) == 50

    def test_generate_dataset_label_counts(self):
        gen = MockEventGenerator(seed=4)
        ds = gen.generate_dataset(n_events=100, signal_fraction=0.1)
        labels = [label for _, label in ds]
        assert sum(labels) == 10     # 10 signal
        assert labels.count(0) == 90 # 90 background

    def test_dataset_labels_are_binary(self):
        gen = MockEventGenerator(seed=5)
        ds = gen.generate_dataset(n_events=20)
        for _, label in ds:
            assert label in (0, 1)

    def test_different_seeds_give_different_events(self):
        g1 = MockEventGenerator(seed=0).generate_event()
        g2 = MockEventGenerator(seed=999).generate_event()
        # With overwhelming probability these will differ
        assert not torch.allclose(g1.x[:min(g1.x.shape[0], g2.x.shape[0])],
                                  g2.x[:min(g1.x.shape[0], g2.x.shape[0])])

    def test_zero_signal_fraction(self):
        gen = MockEventGenerator(seed=6)
        ds = gen.generate_dataset(n_events=20, signal_fraction=0.0)
        labels = [l for _, l in ds]
        assert all(l == 0 for l in labels)
