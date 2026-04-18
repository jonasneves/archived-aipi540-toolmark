"""Train + evaluate the three models. Writes results/scores.json.

Models:
    naive       majority class + keyword heuristic
    classical   XGBoost over TF-IDF + tool one-hot
    deep        DistilBERT fine-tune with tool-name token prepend

Evaluation:
    per-tool recall at 1% FPR, PR-AUC, expected calibration error, on WASP
    held-out set. Includes an ablation with tool-identity features masked.

TODO(Sat–Sun): implement per timeline.
"""

from __future__ import annotations


def main() -> None:
    raise NotImplementedError("implemented on Sat–Sun Apr 18–19; see SCOPE.md")


if __name__ == "__main__":
    main()
