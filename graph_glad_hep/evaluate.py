"""
evaluate.py
-----------
HEP-specific evaluation metrics for anomaly detection at the LHC.

Standard metrics (AUC, AP) plus HEP-specific:
  - Significance Improvement Characteristic (SIC) curve
  - Maximum Improvement in Significance (IS) metric
"""
from __future__ import annotations

import numpy as np
from sklearn.metrics import roc_auc_score, roc_curve, average_precision_score


def sic_curve(
    scores: np.ndarray,
    labels: np.ndarray,
    n_thresholds: int = 200,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Significance Improvement Characteristic (SIC) curve.

    At each threshold t, computes:
      eps_S = fraction of signal with score > t
      eps_B = fraction of background with score > t
      SIC   = eps_S / sqrt(eps_B)

    Args:
        scores:       Anomaly scores, higher = more anomalous.
        labels:       Binary labels (1=signal, 0=background).
        n_thresholds: Number of threshold points.

    Returns:
        (sig_eff, bkg_eff, sic_values) each of shape (n_thresholds,).
    """
    thresholds = np.linspace(scores.min(), scores.max(), n_thresholds)
    sig_eff, bkg_eff, sic_vals = [], [], []

    n_sig = labels.sum() + 1e-12
    n_bkg = (1 - labels).sum() + 1e-12

    for t in thresholds:
        mask = scores > t
        eps_s = labels[mask].sum() / n_sig
        eps_b = (1 - labels)[mask].sum() / n_bkg
        sig_eff.append(float(eps_s))
        bkg_eff.append(float(eps_b))
        sic_vals.append(float(eps_s / np.sqrt(max(eps_b, 1e-12))))

    return np.array(sig_eff), np.array(bkg_eff), np.array(sic_vals)


def evaluate(
    scores: np.ndarray,
    labels: np.ndarray,
) -> dict[str, float | np.ndarray]:
    """Compute full evaluation suite for a set of anomaly scores.

    Args:
        scores: Anomaly scores (N,), higher = more anomalous.
        labels: Binary labels (N,), 1=signal 0=background.

    Returns:
        Dictionary with keys:
          AUC       -- float
          AP        -- float (average precision)
          max_SIC   -- float (peak of SIC curve)
          IS        -- float (max signal_eff / sqrt(bkg_eff) from ROC)
          sig_eff   -- array (SIC curve x-axis)
          bkg_eff   -- array (SIC curve denominator)
          sic       -- array (SIC curve values)
    """
    auc = float(roc_auc_score(labels, scores))
    ap = float(average_precision_score(labels, scores))

    fpr, tpr, _ = roc_curve(labels, scores)
    is_metric = float((tpr / np.sqrt(fpr + 1e-12)).max())

    sig_eff, bkg_eff, sic_vals = sic_curve(scores, labels)

    return {
        "AUC": auc,
        "AP": ap,
        "max_SIC": float(sic_vals.max()),
        "IS": is_metric,
        "sig_eff": sig_eff,
        "bkg_eff": bkg_eff,
        "sic": sic_vals,
    }


def print_results(results: dict) -> None:
    """Pretty-print evaluation results."""
    print(f"  AUC:     {results['AUC']:.4f}")
    print(f"  AP:      {results['AP']:.4f}")
    print(f"  max SIC: {results['max_SIC']:.4f}")
    print(f"  IS:      {results['IS']:.4f}")
