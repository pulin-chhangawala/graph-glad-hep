"""
tests/test_encoder.py
"""
import torch
import pytest
from torch_geometric.data import Data, Batch

from graph_glad_hep.graph_builder import MockEventGenerator
from graph_glad_hep.encoder import GraphEncoder, DualEncoder


def _make_batch(n_graphs=4, n_particles=15, in_dim=5):
    """Create a small synthetic PyG batch for encoder tests."""
    gen = MockEventGenerator(seed=42)
    graphs = [gen.generate_event() for _ in range(n_graphs)]
    return Batch.from_data_list(graphs)


class TestGraphEncoder:

    def test_output_shapes(self):
        batch = _make_batch(n_graphs=4)
        enc = GraphEncoder(in_dim=5, hidden=32, out_dim=16)
        z_node, z_g = enc(batch.x, batch.edge_index, batch.batch)
        assert z_node.shape[1] == 16
        assert z_g.shape == (4, 32)   # 2 * out_dim (mean + max)

    def test_node_count_matches_input(self):
        batch = _make_batch(n_graphs=4)
        enc = GraphEncoder(in_dim=5, hidden=32, out_dim=16)
        z_node, _ = enc(batch.x, batch.edge_index, batch.batch)
        assert z_node.shape[0] == batch.x.shape[0]

    def test_gradients_flow(self):
        batch = _make_batch(n_graphs=2)
        enc = GraphEncoder(in_dim=5, hidden=16, out_dim=8)
        z_node, z_g = enc(batch.x, batch.edge_index, batch.batch)
        z_g.sum().backward()
        for p in enc.parameters():
            assert p.grad is not None

    def test_output_finite(self):
        batch = _make_batch()
        enc = GraphEncoder(in_dim=5, hidden=32, out_dim=16)
        z_node, z_g = enc(batch.x, batch.edge_index, batch.batch)
        assert torch.isfinite(z_node).all()
        assert torch.isfinite(z_g).all()


class TestDualEncoder:

    def test_output_tuple_length(self):
        batch = _make_batch(n_graphs=4)
        model = DualEncoder(in_dim=5, hidden=32, out_dim=16)
        out = model(batch.x, batch.edge_index, batch.batch)
        assert len(out) == 7   # z_node, z_g, zhat_node, zhat_g, h_g, hhat_g, x_hat

    def test_projection_shape(self):
        batch = _make_batch(n_graphs=4)
        model = DualEncoder(in_dim=5, hidden=32, out_dim=16)
        _, _, _, _, h_g, hhat_g, _ = model(batch.x, batch.edge_index, batch.batch)
        assert h_g.shape == (4, 64)
        assert hhat_g.shape == (4, 64)

    def test_x_hat_shape_matches_x(self):
        batch = _make_batch(n_graphs=3)
        model = DualEncoder(in_dim=5, hidden=32, out_dim=16)
        *_, x_hat = model(batch.x, batch.edge_index, batch.batch)
        assert x_hat.shape == batch.x.shape

    def test_perturbed_differs_from_clean(self):
        batch = _make_batch(n_graphs=4)
        model = DualEncoder(in_dim=5, hidden=32, out_dim=16, eta=2.0)
        _, z_g, _, zhat_g, _, _, _ = model(batch.x, batch.edge_index, batch.batch)
        # With eta=2.0 the perturbed branch should differ
        assert not torch.allclose(z_g, zhat_g)

    def test_all_outputs_finite(self):
        batch = _make_batch(n_graphs=4)
        model = DualEncoder(in_dim=5, hidden=32, out_dim=16)
        outputs = model(batch.x, batch.edge_index, batch.batch)
        for t in outputs:
            assert torch.isfinite(t).all()
