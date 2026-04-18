"""Train and evaluate the three models under Leave-One-Tool-Class-Out.

Models:
    naive       majority class + keyword heuristic (tool-agnostic)
    classical   XGBoost over char n-gram TF-IDF + tool-class one-hot
    deep        DistilBERT fine-tune with `[TOOL: <class>]` prepend

For each of the 8 tool classes, the held-out class forms the test fold;
the remaining 7 classes form train + validation. The threshold that
achieves FPR ≤ 1% is calibrated on the validation slice and applied
verbatim to the held-out class.

Sources used:
    HuggingFace transformers (Apache 2.0) — model training loop
    https://github.com/huggingface/transformers
    scikit-learn (BSD 3-clause) — vectorizers, metrics
    XGBoost (Apache 2.0) — classical model
"""

from __future__ import annotations

import argparse
import json
import logging
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import numpy as np
import pandas as pd
import torch
import xgboost as xgb
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset as TorchDataset
from transformers import (
    AutoTokenizer,
    DistilBertForSequenceClassification,
    Trainer,
    TrainingArguments,
)

from scripts import build_features, metrics as metrics_mod

log = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parent.parent
PROCESSED_PATH = REPO_ROOT / "data" / "processed" / "toolmark.parquet"
RESULTS_PATH = REPO_ROOT / "results" / "scores.json"
MODELS_DIR = REPO_ROOT / "models"

DISTILBERT_CHECKPOINT = "distilbert-base-uncased"
SEED = 540

# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------


class Classifier(Protocol):
    """Common interface across naive, classical, and deep models."""

    name: str

    def fit(self, train_df: pd.DataFrame, val_df: pd.DataFrame) -> None: ...

    def predict_proba(self, df: pd.DataFrame) -> np.ndarray:
        """Return malicious-class probabilities in [0, 1]."""


# Keyword heuristic tokens for NaiveModel. Chosen from indirect-prompt-
# injection literature: imperative phrasing and exfiltration verbs.
INJECTION_KEYWORDS: tuple[str, ...] = (
    "ignore previous",
    "ignore the above",
    "ignore all prior",
    "disregard",
    "new instructions",
    "system prompt",
    "send to",
    "forward to",
    "email to",
    "transfer to",
    "grant access",
    "delete all",
    "execute",
    "curl http",
    "please",
)

KEYWORD_REGEX = re.compile("|".join(re.escape(k) for k in INJECTION_KEYWORDS), re.IGNORECASE)


@dataclass
class NaiveModel:
    """Majority-class prior blended with a lightweight keyword score.

    The keyword score is the fraction of the text matched by the injection
    keyword regex. On balanced Toolmark, the majority prior is 0.5 and the
    keyword signal dominates — which is the point: a classifier with no
    representation of the tool is already sensitive to obvious cues.
    """

    name: str = "naive_keyword"

    def fit(self, train_df: pd.DataFrame, val_df: pd.DataFrame) -> None:
        self._majority = float(train_df["label"].mean())

    def predict_proba(self, df: pd.DataFrame) -> np.ndarray:
        scores = np.zeros(len(df), dtype=float)
        for i, text in enumerate(df["text"].tolist()):
            matches = len(KEYWORD_REGEX.findall(text))
            length = max(len(text.split()), 1)
            scores[i] = min(1.0, self._majority + matches / length * 4.0)
        return scores


@dataclass
class ClassicalModel:
    """XGBoost over char n-gram TF-IDF (+ optional tool-class one-hot)."""

    include_tool_feature: bool = True
    name: str = "classical_xgb"

    def fit(self, train_df: pd.DataFrame, val_df: pd.DataFrame) -> None:
        self._featurizer = build_features.fit_featurizer(
            train_df, include_tool_feature=self.include_tool_feature
        )
        x_train = self._featurizer.transform(train_df)
        y_train = train_df["label"].values
        self._model = xgb.XGBClassifier(
            n_estimators=300,
            max_depth=6,
            learning_rate=0.1,
            subsample=0.9,
            colsample_bytree=0.9,
            eval_metric="logloss",
            tree_method="hist",
            n_jobs=1,  # avoid macOS fork+openmp crashes; dataset is small
            random_state=SEED,
        )
        self._model.fit(x_train, y_train)

    def predict_proba(self, df: pd.DataFrame) -> np.ndarray:
        x = self._featurizer.transform(df)
        return self._model.predict_proba(x)[:, 1]


# ---------------------------------------------------------------------------
# DistilBERT fine-tuning
# ---------------------------------------------------------------------------


class _TextDataset(TorchDataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx):
        item = {k: torch.as_tensor(v[idx]) for k, v in self.encodings.items()}
        item["labels"] = torch.as_tensor(self.labels[idx])
        return item


def _pick_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


@dataclass
class DeepModel:
    """DistilBERT fine-tune with `[TOOL: <class>]` prepend in the input text."""

    include_tool_feature: bool = True
    max_length: int = 256
    epochs: int = 2
    per_device_batch: int = 16
    name: str = "deep_distilbert"

    def _encode(self, df: pd.DataFrame, tokenizer):
        texts = build_features.distilbert_input_texts(
            df, include_tool_feature=self.include_tool_feature
        )
        return tokenizer(
            texts, padding=True, truncation=True, max_length=self.max_length, return_tensors="np"
        )

    def fit(self, train_df: pd.DataFrame, val_df: pd.DataFrame) -> None:
        tokenizer = AutoTokenizer.from_pretrained(DISTILBERT_CHECKPOINT)
        model = DistilBertForSequenceClassification.from_pretrained(
            DISTILBERT_CHECKPOINT, num_labels=2
        )
        device = _pick_device()
        log.info("distilbert training on device=%s", device)

        train_ds = _TextDataset(self._encode(train_df, tokenizer), train_df["label"].values)
        val_ds = _TextDataset(self._encode(val_df, tokenizer), val_df["label"].values)

        tmpdir = MODELS_DIR / ".distilbert_tmp"
        tmpdir.mkdir(parents=True, exist_ok=True)
        args = TrainingArguments(
            output_dir=str(tmpdir),
            num_train_epochs=self.epochs,
            per_device_train_batch_size=self.per_device_batch,
            per_device_eval_batch_size=self.per_device_batch,
            learning_rate=5e-5,
            weight_decay=0.01,
            logging_steps=50,
            save_strategy="no",
            report_to=[],
            seed=SEED,
            use_cpu=(device == "cpu"),
        )
        trainer = Trainer(model=model, args=args, train_dataset=train_ds, eval_dataset=val_ds)
        trainer.train()

        self._model = model
        self._tokenizer = tokenizer
        self._device = device

    def predict_proba(self, df: pd.DataFrame) -> np.ndarray:
        enc = self._encode(df, self._tokenizer)
        self._model.eval()
        self._model.to(self._device)

        input_ids = torch.as_tensor(enc["input_ids"]).to(self._device)
        attention_mask = torch.as_tensor(enc["attention_mask"]).to(self._device)
        with torch.no_grad():
            logits = self._model(input_ids=input_ids, attention_mask=attention_mask).logits
            probs = torch.softmax(logits, dim=-1)[:, 1].cpu().numpy()
        return probs


# ---------------------------------------------------------------------------
# LOTCO training + evaluation loop
# ---------------------------------------------------------------------------


def split_train_val(df: pd.DataFrame, val_frac: float = 0.15) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Template-aware split so matched (benign, malicious) pairs stay together."""
    # np.asarray() escapes pandas' pyarrow-backed extension arrays, which
    # sklearn's train_test_split cannot slice directly.
    template_ids = np.asarray(df["template_id"].unique(), dtype=object)
    train_ids, val_ids = train_test_split(template_ids, test_size=val_frac, random_state=SEED)
    train_df = df[df["template_id"].isin(train_ids)].reset_index(drop=True)
    val_df = df[df["template_id"].isin(val_ids)].reset_index(drop=True)
    return train_df, val_df


def _eval_one(
    model_factory, train_df: pd.DataFrame, val_df: pd.DataFrame, test_df: pd.DataFrame
) -> metrics_mod.Scores:
    t0 = time.time()
    model = model_factory()
    log.info("  fitting %s  (train=%d val=%d)", model.name, len(train_df), len(val_df))
    model.fit(train_df, val_df)

    val_scores = model.predict_proba(val_df)
    y_val = val_df["label"].values
    threshold = metrics_mod.threshold_for_fpr(y_val, val_scores, max_fpr=0.01)

    test_scores = model.predict_proba(test_df)
    y_test = test_df["label"].values
    result = metrics_mod.score(y_test, test_scores, threshold)
    log.info(
        "  %s  recall@1%%FPR=%.3f  PR-AUC=%.3f  ROC-AUC=%.3f  ECE=%.3f  (%.1fs)",
        model.name,
        result.recall_at_1pct_fpr,
        result.pr_auc,
        result.roc_auc,
        result.ece,
        time.time() - t0,
    )
    return result


def run_lotco(df: pd.DataFrame, deep: bool = True) -> dict:
    """Leave-One-Tool-Class-Out sweep across all 8 classes × 3 models."""
    results: dict[str, dict[str, dict[str, float]]] = {}
    tool_classes = sorted(df["tool_class"].unique())
    log.info("LOTCO over %d tool classes: %s", len(tool_classes), tool_classes)

    model_factories = {
        "naive_keyword": lambda: NaiveModel(),
        "classical_xgb": lambda: ClassicalModel(include_tool_feature=True),
        "classical_xgb_no_tool": lambda: ClassicalModel(include_tool_feature=False),
    }
    if deep:
        model_factories["deep_distilbert"] = lambda: DeepModel(include_tool_feature=True)
        model_factories["deep_distilbert_no_tool"] = lambda: DeepModel(include_tool_feature=False)

    for held_out in tool_classes:
        log.info("=== held-out tool class: %s ===", held_out)
        test_df = df[df["tool_class"] == held_out].reset_index(drop=True)
        pool_df = df[df["tool_class"] != held_out].reset_index(drop=True)
        train_df, val_df = split_train_val(pool_df)
        log.info(
            "  test=%d train=%d val=%d (7 remaining tool classes)",
            len(test_df),
            len(train_df),
            len(val_df),
        )

        results[held_out] = {}
        for model_name, factory in model_factories.items():
            scores = _eval_one(factory, train_df, val_df, test_df)
            results[held_out][model_name] = scores.as_dict()

    return results


# ---------------------------------------------------------------------------
# Aggregation + main
# ---------------------------------------------------------------------------


def aggregate(per_class: dict) -> dict[str, dict[str, float]]:
    """Mean metric per model across all held-out classes."""
    summary: dict[str, dict[str, float]] = {}
    for held_out, per_model in per_class.items():
        for model, metrics in per_model.items():
            summary.setdefault(model, {"recall_at_1pct_fpr": 0.0, "pr_auc": 0.0, "roc_auc": 0.0, "ece": 0.0, "n_folds": 0})
            for key in ("recall_at_1pct_fpr", "pr_auc", "roc_auc", "ece"):
                value = metrics[key]
                if value is not None and not (isinstance(value, float) and np.isnan(value)):
                    summary[model][key] += value
            summary[model]["n_folds"] += 1
    for model, totals in summary.items():
        n = totals["n_folds"]
        for key in ("recall_at_1pct_fpr", "pr_auc", "roc_auc", "ece"):
            totals[key] = totals[key] / n if n else float("nan")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="LOTCO train + eval harness")
    parser.add_argument("--no-deep", action="store_true", help="skip DistilBERT to iterate quickly")
    args = parser.parse_args()

    # force=True so we override any handlers installed by transformers/torch
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        force=True,
    )
    print("[toolmark] starting LOTCO run", flush=True)
    df = pd.read_parquet(PROCESSED_PATH)
    log.info("loaded %d rows from %s", len(df), PROCESSED_PATH)

    per_class = run_lotco(df, deep=not args.no_deep)
    summary = aggregate(per_class)

    RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "per_class": per_class,
        "summary": summary,
        "config": {
            "seed": SEED,
            "distilbert_checkpoint": DISTILBERT_CHECKPOINT,
            "ran_deep": not args.no_deep,
        },
    }
    with RESULTS_PATH.open("w") as f:
        json.dump(payload, f, indent=2, default=float)
    log.info("wrote %s", RESULTS_PATH)


if __name__ == "__main__":
    main()
