"""
tests/test_losses_and_decorrelation.py
"""
import numpy as np
import pytest
import torch


# ---------------------------------------------------------------------------
# contrastive_loss
# ---------------------------------------------------------------------------

class TestContrastiveLoss:

    def test_perfect_alignment_low_loss(self):
        from graph_glad_hep.losses import contrastive_loss
        B, D = 8, 32
        h = torch.randn(B, D)
        # perturbed = clean + tiny noise => should give low loss
        h_hat = h + 1e-5 * torch.randn_like(h)
        loss = contrastive_loss(h, h_hat, temperature=0.2)
        assert loss.item() < 2.0   # log(B) = 2.08 is the random baseline

    def test_random_embeddings_high_loss(self):
        from graph_glad_hep.losses import contrastive_loss
        B, D = 16, 64
        h = torch.randn(B, D)
        h_hat = torch.randn(B, D)
        loss = contrastive_loss(h, h_hat, temperature=0.2)
        assert loss.item() > 0.0

    def test_loss_is_scalar(self):
        from graph_glad_hep.losses import contrastive_loss
        h = torch.randn(4, 16)
        h_hat = torch.randn(4, 16)
        loss = contrastive_loss(h, h_hat)
        assert loss.shape == torch.Size([])

    def test_loss_is_finite(self):
        from graph_glad_hep.losses import contrastive_loss
        h = torch.randn(8, 32)
        h_hat = torch.randn(8, 32)
        assert torch.isfinite(contrastive_loss(h, h_hat))

    def test_gradient_flows(self):
        from graph_glad_hep.losses import contrastive_loss
        h = torch.randn(4, 16, requires_grad=True)
        h_hat = torch.randn(4, 16, requires_grad=True)
        loss = contrastive_loss(h, h_hat)
        loss.backward()
        assert h.grad is not None
        assert h_hat.grad is not None


# ---------------------------------------------------------------------------
# dist_corr
# ---------------------------------------------------------------------------

class TestDistCorr:

    def test_independent_variables_near_zero(self):
        from graph_glad_hep.decorrelation import dist_corr
        torch.manual_seed(0)
        # Large sample of truly independent Gaussians
        X = torch.randn(200)
        Y = torch.randn(200)
        dc = dist_corr(X, Y)
        # Not exactly 0, but should be small
        assert dc.item() < 0.3

    def test_identical_variables_near_one(self):
        from graph_glad_hep.decorrelation import dist_corr
        X = torch.linspace(0, 1, 50)
        Y = X.clone()
        dc = dist_corr(X, Y)
        assert dc.item() > 0.9

    def test_output_in_unit_interval(self):
        from graph_glad_hep.decorrelation import dist_corr
        X = torch.randn(30)
        Y = torch.randn(30)
        dc = dist_corr(X, Y)
        assert 0.0 <= dc.item() <= 1.0 + 1e-6

    def test_gradient_flows_through_scores(self):
        from graph_glad_hep.decorrelation import dist_corr
        X = torch.randn(20, requires_grad=True)
        Y = torch.randn(20)
        dc = dist_corr(X, Y)
        dc.backward()
        assert X.grad is not None

    def test_output_is_scalar(self):
        from graph_glad_hep.decorrelation import dist_corr
        dc = dist_corr(torch.randn(10), torch.randn(10))
        assert dc.shape == torch.Size([])

    def test_single_element(self):
        from graph_glad_hep.decorrelation import dist_corr
        dc = dist_corr(torch.tensor([1.0]), torch.tensor([2.0]))
        assert dc.item() == pytest.approx(0.0, abs=1e-5)


# ---------------------------------------------------------------------------
# evaluate
# ---------------------------------------------------------------------------

class TestEvaluate:

    def _make_scores(self, n=200):
        rng = np.random.default_rng(0)
        labels = np.array([0] * 180 + [1] * 20)
        # signal has higher scores
        scores = rng.normal(0.3, 0.1, n)
        scores[labels == 1] += 0.5
        return scores, labels

    def test_evaluate_returns_dict(self):
        from graph_glad_hep.evaluate import evaluate
        scores, labels = self._make_scores()
        res = evaluate(scores, labels)
        assert isinstance(res, dict)

    def test_evaluate_has_all_keys(self):
        from graph_glad_hep.evaluate import evaluate
        scores, labels = self._make_scores()
        res = evaluate(scores, labels)
        for key in ("AUC", "AP", "max_SIC", "IS", "sig_eff", "bkg_eff", "sic"):
            assert key in res, f"Missing key: {key}"

    def test_auc_above_random(self):
        from graph_glad_hep.evaluate import evaluate
        scores, labels = self._make_scores()
        res = evaluate(scores, labels)
        assert res["AUC"] > 0.5

    def test_sic_curve_length(self):
        from graph_glad_hep.evaluate import evaluate
        scores, labels = self._make_scores()
        res = evaluate(scores, labels)
        assert len(res["sic"]) == 200

    def test_max_sic_positive(self):
        from graph_glad_hep.evaluate import evaluate
        scores, labels = self._make_scores()
        res = evaluate(scores, labels)
        assert res["max_SIC"] > 0.0

    def test_sic_curve_standalone(self):
        from graph_glad_hep.evaluate import sic_curve
        scores, labels = self._make_scores()
        sig_eff, bkg_eff, sic = sic_curve(scores, labels, n_thresholds=50)
        assert len(sig_eff) == 50
        assert all(0 <= v <= 1.0 + 1e-6 for v in sig_eff)
        assert all(v >= 0 for v in sic)
