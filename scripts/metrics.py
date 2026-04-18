"""Evaluation metrics for the Toolmark classifier.

Primary metric is **recall at 1% FPR**, chosen because an in-line
guardrail's operating point matters more than any threshold-invariant
score: false positives block benign tool outputs and destroy UX, false
negatives let injection through.

The threshold that achieves FPR ≤ 0.01 is calibrated on a held-out
validation slice (not on the test fold), then applied verbatim to the
test fold — no test-set peeking.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score, roc_curve


@dataclass(frozen=True)
class Scores:
    recall_at_1pct_fpr: float
    threshold_at_1pct_fpr: float
    pr_auc: float
    roc_auc: float
    ece: float
    n_pos: int
    n_neg: int

    def as_dict(self) -> dict[str, float]:
        return asdict(self)


def threshold_for_fpr(y_val: np.ndarray, val_scores: np.ndarray, max_fpr: float = 0.01) -> float:
    """Smallest threshold on the val set whose FPR does not exceed max_fpr.

    Falls back to the maximum observed score + epsilon if no threshold can
    hit the requested FPR (e.g., when the validation split has zero
    negatives, which should not happen in this pipeline but is guarded for).
    """
    fpr, _tpr, thresholds = roc_curve(y_val, val_scores)
    eligible = np.where(fpr <= max_fpr)[0]
    if eligible.size == 0:
        return float(np.max(val_scores)) + 1e-9
    return float(thresholds[eligible[-1]])


def recall_at_threshold(y_true: np.ndarray, scores: np.ndarray, threshold: float) -> float:
    pos = y_true == 1
    if pos.sum() == 0:
        return float("nan")
    predicted_pos = scores >= threshold
    return float((predicted_pos & pos).sum() / pos.sum())


def expected_calibration_error(y_true: np.ndarray, scores: np.ndarray, n_bins: int = 10) -> float:
    """Bin-based ECE. Reports weighted mean of |bin_accuracy - bin_confidence|."""
    bin_edges = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    n = len(y_true)
    if n == 0:
        return float("nan")
    for lo, hi in zip(bin_edges[:-1], bin_edges[1:]):
        mask = (scores >= lo) & (scores < hi) if hi < 1 else (scores >= lo) & (scores <= hi)
        if mask.sum() == 0:
            continue
        bin_accuracy = float(y_true[mask].mean())
        bin_confidence = float(scores[mask].mean())
        ece += (mask.sum() / n) * abs(bin_accuracy - bin_confidence)
    return ece


def score(y_true: np.ndarray, scores: np.ndarray, threshold: float) -> Scores:
    """Compute the full score bundle for one (model, held-out-class) pair."""
    recall = recall_at_threshold(y_true, scores, threshold)
    return Scores(
        recall_at_1pct_fpr=recall,
        threshold_at_1pct_fpr=float(threshold),
        pr_auc=float(average_precision_score(y_true, scores)) if len(set(y_true)) > 1 else float("nan"),
        roc_auc=float(roc_auc_score(y_true, scores)) if len(set(y_true)) > 1 else float("nan"),
        ece=expected_calibration_error(y_true, scores),
        n_pos=int((y_true == 1).sum()),
        n_neg=int((y_true == 0).sum()),
    )
