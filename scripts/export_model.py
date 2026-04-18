"""Train the final DistilBERT on the full Toolmark corpus and export to ONNX.

This is the "deployment" model — trained on all 2,108 rows rather than per
LOTCO fold, then exported to an ONNX graph that `transformers.js` can load
and run in the browser via ONNX Runtime Web + WebGPU.

Writes:
    models/toolmark_distilbert/        the trained PyTorch model + tokenizer
    public/models/toolmark-distilbert/ ONNX graph + tokenizer for the web app

Sources used:
    HuggingFace Optimum (Apache 2.0) — ONNX export pipeline
    https://github.com/huggingface/optimum
"""

from __future__ import annotations

import logging
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset as TorchDataset
from transformers import (
    AutoTokenizer,
    DistilBertForSequenceClassification,
    Trainer,
    TrainingArguments,
)

from scripts.build_features import distilbert_input_texts
from scripts.model import DISTILBERT_CHECKPOINT, SEED, _pick_device

log = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parent.parent
PROCESSED_PATH = REPO_ROOT / "data" / "processed" / "toolmark.parquet"
PT_OUT = REPO_ROOT / "models" / "toolmark_distilbert"
ONNX_OUT = REPO_ROOT / "public" / "models" / "toolmark-distilbert"


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


def _encode(df: pd.DataFrame, tokenizer, max_length: int = 192):
    texts = distilbert_input_texts(df, include_tool_feature=True)
    return tokenizer(texts, padding=True, truncation=True, max_length=max_length, return_tensors="np")


def train_full() -> None:
    df = pd.read_parquet(PROCESSED_PATH)
    log.info("loaded %d rows", len(df))

    # Template-aware val split so matched pairs stay together
    template_ids = np.asarray(df["template_id"].unique(), dtype=object)
    train_ids, val_ids = train_test_split(template_ids, test_size=0.1, random_state=SEED)
    train_df = df[df["template_id"].isin(train_ids)].reset_index(drop=True)
    val_df = df[df["template_id"].isin(val_ids)].reset_index(drop=True)
    log.info("train=%d val=%d", len(train_df), len(val_df))

    tokenizer = AutoTokenizer.from_pretrained(DISTILBERT_CHECKPOINT)
    model = DistilBertForSequenceClassification.from_pretrained(DISTILBERT_CHECKPOINT, num_labels=2)

    device = _pick_device()
    log.info("training on device=%s", device)

    train_ds = _TextDataset(_encode(train_df, tokenizer), train_df["label"].values)
    val_ds = _TextDataset(_encode(val_df, tokenizer), val_df["label"].values)

    PT_OUT.mkdir(parents=True, exist_ok=True)
    args = TrainingArguments(
        output_dir=str(PT_OUT / ".tmp"),
        num_train_epochs=2,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
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

    model.save_pretrained(PT_OUT)
    tokenizer.save_pretrained(PT_OUT)
    log.info("saved PyTorch model to %s", PT_OUT)


def export_onnx() -> None:
    """Use optimum-cli to export the saved PyTorch model to ONNX."""
    import subprocess

    if ONNX_OUT.exists():
        shutil.rmtree(ONNX_OUT)
    ONNX_OUT.mkdir(parents=True)
    cmd = [
        "optimum-cli",
        "export",
        "onnx",
        "--model",
        str(PT_OUT),
        "--task",
        "text-classification",
        str(ONNX_OUT),
    ]
    log.info("running: %s", " ".join(cmd))
    subprocess.run(cmd, check=True)
    log.info("exported ONNX to %s", ONNX_OUT)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s", force=True)
    print("[toolmark] training final DistilBERT on full corpus", flush=True)
    train_full()
    print("[toolmark] exporting to ONNX", flush=True)
    export_onnx()
    print("[toolmark] done — model ready at public/models/toolmark-distilbert/", flush=True)


if __name__ == "__main__":
    main()
