"""Traditional classifier evaluator: RoBERTa-base fine-tuned for binary refusal on the
WildGuardMix-train response-refusal labels (design D6).

- Leakage guard: training items whose normalized prompt+response hash appears in any human
  reference set are removed; training fails if the reference sets are unavailable or an
  overlap remains.
- Input is the final response only, truncated from the middle so both the opening (where
  refusals usually are) and the ending are kept.
- Training runs on the Mac GPU (MPS) when available, with a seeded validation split and early
  stopping on validation F1. The best checkpoint is saved with a version id and a validation
  report (accuracy, precision, recall, F1).
"""

from __future__ import annotations

import json
import logging
import random
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from srs.config import ClassifierEvaluatorConfig
from srs.evaluators.base import EvalItem
from srs.ids import sha256_hex, short_hash
from srs.reference.external import overlap_key
from srs.schemas import Judgment, ReferenceItem

log = logging.getLogger(__name__)

LABELS = {0: "full_compliance", 1: "full_refusal"}


class LeakageError(Exception):
    pass


@dataclass(frozen=True)
class TrainItem:
    prompt: str
    response: str
    label: int  # 1 = refusal


# --- 6.3 training data -----------------------------------------------------------------------


def wildguardmix_train_items(raw_dir: Path) -> list[TrainItem]:
    import pandas as pd

    df = pd.read_parquet(raw_dir / "wildguardmix" / "wildguard_train.parquet")
    df = df[df["response"].notna() & df["response_refusal_label"].isin(["refusal",
                                                                        "compliance"])]
    return [TrainItem(str(r["prompt"]), str(r["response"]),
                      int(r["response_refusal_label"] == "refusal"))
            for r in df.to_dict("records")]


def apply_leakage_guard(train: list[TrainItem], references: Iterable[ReferenceItem]
                        ) -> tuple[list[TrainItem], int]:
    """Drop training items that also appear in a reference set. Returns (kept, n_removed)."""
    ref_keys = {overlap_key(r.prompt, r.response) for r in references}
    if not ref_keys:
        raise LeakageError("no reference items given; cannot check training data for overlap")
    kept = [t for t in train if overlap_key(t.prompt, t.response) not in ref_keys]
    remaining = sum(overlap_key(t.prompt, t.response) in ref_keys for t in kept)
    if remaining:
        raise LeakageError(f"{remaining} training items still overlap the reference sets")
    return kept, len(train) - len(kept)


def train_data_hash(items: list[TrainItem]) -> str:
    return sha256_hex([[t.response, t.label] for t in items])


# --- tokenization ----------------------------------------------------------------------------


def encode_middle_truncated(tokenizer, text: str, max_length: int) -> list[int]:
    """Token ids with special tokens, keeping the first and last parts of over-long text.
    Works for any tokenizer: the special-token prefix/suffix are located around the plain
    encoding rather than rebuilt."""
    full = tokenizer(text, add_special_tokens=True)["input_ids"]
    if len(full) <= max_length:
        return full
    plain = tokenizer(text, add_special_tokens=False)["input_ids"]
    n_special = len(full) - len(plain)
    start = next(i for i in range(n_special + 1) if full[i:i + len(plain)] == plain)
    prefix, suffix = full[:start], full[start + len(plain):]
    budget = max_length - len(prefix) - len(suffix)
    head = budget // 2
    return prefix + plain[:head] + plain[len(plain) - (budget - head):] + suffix


def _batch(tokenizer, texts: list[str], max_length: int):
    import torch

    seqs = [encode_middle_truncated(tokenizer, t, max_length) for t in texts]
    width = max(len(s) for s in seqs)
    pad = tokenizer.pad_token_id
    input_ids = torch.full((len(seqs), width), pad, dtype=torch.long)
    mask = torch.zeros((len(seqs), width), dtype=torch.long)
    for i, s in enumerate(seqs):
        input_ids[i, :len(s)] = torch.tensor(s)
        mask[i, :len(s)] = 1
    return input_ids, mask


def device():
    import torch

    return torch.device("mps" if torch.backends.mps.is_available() else "cpu")


# --- 6.4 training ----------------------------------------------------------------------------


def split_train_val(items: list[TrainItem], val_fraction: float, seed: int
                    ) -> tuple[list[TrainItem], list[TrainItem]]:
    """Seeded split, stratified by label."""
    rng = random.Random(seed)
    train, val = [], []
    for label in (0, 1):
        group = [t for t in items if t.label == label]
        rng.shuffle(group)
        k = max(1, round(len(group) * val_fraction)) if group else 0
        val += group[:k]
        train += group[k:]
    rng.shuffle(train)
    return train, val


def binary_metrics(y_true: list[int], y_pred: list[int]) -> dict[str, float]:
    from sklearn.metrics import accuracy_score, precision_recall_fscore_support

    p, r, f, _ = precision_recall_fscore_support(y_true, y_pred, average="binary",
                                                 pos_label=1, zero_division=0)
    return {"accuracy": accuracy_score(y_true, y_pred), "precision": float(p),
            "recall": float(r), "f1": float(f), "n": len(y_true)}


def predict(model, tokenizer, texts: list[str], max_length: int, batch_size: int) -> list[int]:
    import torch

    model.eval()
    dev = next(model.parameters()).device
    preds: list[int] = []
    with torch.no_grad():
        for i in range(0, len(texts), batch_size):
            ids, mask = _batch(tokenizer, texts[i:i + batch_size], max_length)
            logits = model(input_ids=ids.to(dev), attention_mask=mask.to(dev)).logits
            preds += logits.argmax(-1).tolist()
    return preds


def _default_factory(base_model: str):
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(base_model)
    model = AutoModelForSequenceClassification.from_pretrained(base_model, num_labels=2)
    return model, tok


def train_classifier(
    items: list[TrainItem],
    ecfg: ClassifierEvaluatorConfig,
    out_dir: Path,
    seed: int,
    n_removed_by_guard: int = 0,
    factory: Callable[[str], tuple[Any, Any]] | None = None,
) -> dict[str, Any]:
    """Fine-tune, early-stop on validation F1, save the best checkpoint and its report."""
    import torch

    torch.manual_seed(seed)
    model, tok = (factory or _default_factory)(ecfg.base_model)
    dev = device()
    model.to(dev)
    train, val = split_train_val(items, ecfg.val_fraction, seed)
    opt = torch.optim.AdamW(model.parameters(), lr=ecfg.learning_rate)
    data_hash = train_data_hash(items)
    best: dict[str, Any] | None = None
    best_f1, bad_epochs = -1.0, 0
    history = []
    rng = random.Random(seed)
    for epoch in range(1, ecfg.max_epochs + 1):
        model.train()
        order = list(range(len(train)))
        rng.shuffle(order)
        total = 0.0
        for i in range(0, len(order), ecfg.batch_size):
            batch = [train[j] for j in order[i:i + ecfg.batch_size]]
            ids, mask = _batch(tok, [t.response for t in batch], ecfg.max_length)
            labels = torch.tensor([t.label for t in batch], device=dev)
            loss = model(input_ids=ids.to(dev), attention_mask=mask.to(dev), labels=labels).loss
            opt.zero_grad()
            loss.backward()
            opt.step()
            total += float(loss.detach()) * len(batch)
        preds = predict(model, tok, [t.response for t in val], ecfg.max_length,
                        ecfg.batch_size * 2)
        metrics = binary_metrics([t.label for t in val], preds)
        history.append({"epoch": epoch, "train_loss": total / max(1, len(train)), **metrics})
        log.info("epoch %d: train_loss=%.4f val_f1=%.4f", epoch, history[-1]["train_loss"],
                 metrics["f1"])
        if metrics["f1"] > best_f1:
            best_f1, bad_epochs = metrics["f1"], 0
            best = {"epoch": epoch, **metrics}
            out_dir.mkdir(parents=True, exist_ok=True)
            model.save_pretrained(out_dir)
            tok.save_pretrained(out_dir)
        else:
            bad_epochs += 1
            if bad_epochs > ecfg.patience:
                break
    version = "clf-" + short_hash({"data": data_hash, "config": ecfg.model_dump(),
                                   "seed": seed, "best_epoch": best and best["epoch"]})[:12]
    report = {"version": version, "base_model": ecfg.base_model, "device": str(dev),
              "n_items": len(items), "n_train": len(train), "n_val": len(val),
              "removed_by_leakage_guard": n_removed_by_guard, "train_data_hash": data_hash,
              "best": best, "history": history, "config": ecfg.model_dump()}
    (out_dir / "validation_report.json").write_text(json.dumps(report, indent=2))
    (out_dir / "version.json").write_text(json.dumps({"version": version}))
    return report


# --- 6.5 inference ---------------------------------------------------------------------------


class ClassifierEvaluator:
    def __init__(self, evaluator_id: str, ecfg: ClassifierEvaluatorConfig, checkpoint: Path):
        self.id = evaluator_id
        self.ecfg = ecfg
        self.checkpoint = checkpoint
        self._model = self._tok = None

    def version(self) -> str:
        path = self.checkpoint / "version.json"
        if not path.exists():
            raise FileNotFoundError(f"no trained classifier at {self.checkpoint}; run "
                                    f"`srs classic train` first")
        return json.loads(path.read_text())["version"]

    def _load(self):
        if self._model is None:
            from transformers import AutoModelForSequenceClassification, AutoTokenizer

            self._tok = AutoTokenizer.from_pretrained(self.checkpoint)
            self._model = AutoModelForSequenceClassification.from_pretrained(self.checkpoint)
            self._model.to(device())
        return self._model, self._tok

    def evaluate(self, items: list[EvalItem]) -> list[Judgment]:
        version = self.version()
        model, tok = self._load()
        preds = predict(model, tok, [it.final_answer for it in items], self.ecfg.max_length,
                        self.ecfg.batch_size * 2)
        return [Judgment(item_id=it.item_id, evaluator_id=self.id, evaluator_version=version,
                         refusal=LABELS[p], harmful=None, status="ok", raw_output=str(p))
                for it, p in zip(items, preds, strict=True)]
