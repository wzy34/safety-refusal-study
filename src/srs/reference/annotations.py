"""Self-annotation storage and logic, shared by the Streamlit app and `srs annotate import`.

Files in an annotation directory (results/annotations/<name>/):
  items.parquet          blinded items (from the sampler)
  pass1.parquet          labels, written after every single label
  pass2.parquet
  adjudication.parquet   final labels for items whose two passes disagree
  human_labels.parquet   import output: pass 1, pass 2 and final labels (pass_num 0)
  reliability.json/.md   test-retest kappa and the disagreement list

A single-pass import (pilot calibration) takes the pass-1 labels as final.

Blinding: the harmful question is asked for every item, because asking it only for
refuse-expected prompts would reveal the expected behavior.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import pandas as pd

from srs.ids import sha256_hex
from srs.io import read_records, upsert_records, write_records
from srs.schemas import ADJUDICATED_PASS, HumanLabel

APP_PATH = Path(__file__).with_name("annotate_app.py")

REFUSAL_CHOICES = (("1", "full_compliance", "Full compliance"),
                   ("2", "partial", "Partial refusal"),
                   ("3", "full_refusal", "Full refusal"))


def load_items(d: Path) -> list[dict[str, Any]]:
    df = pd.read_parquet(d / "items.parquet")
    return [{"item_id": r["item_id"], "turns": [dict(t) for t in r["turns"]]}
            for r in df.to_dict("records")]


def label_file(d: Path, pass_num: int) -> Path:
    return d / ("adjudication.parquet" if pass_num == ADJUDICATED_PASS else
                f"pass{pass_num}.parquet")


def load_labels(d: Path, pass_num: int) -> list[HumanLabel]:
    return read_records(label_file(d, pass_num), HumanLabel)


def save_label(d: Path, label: HumanLabel) -> None:
    """Persist one label immediately (a crash loses at most the label being entered)."""
    upsert_records(label_file(d, label.pass_num), HumanLabel, [label], key=lambda x: x.item_id)


def queue(item_ids: list[str], pass_num: int, seed: int) -> list[str]:
    """Presentation order: a seeded shuffle that differs between passes."""
    return sorted(item_ids, key=lambda i: sha256_hex(f"{seed}:pass{pass_num}:{i}"))


def next_unlabeled(order: list[str], labeled: set[str]) -> int | None:
    return next((k for k, i in enumerate(order) if i not in labeled), None)


@dataclass
class Gate:
    open: bool
    message: str = ""
    available_at: datetime | None = None


def _parse(ts: str) -> datetime:
    dt = datetime.fromisoformat(ts)
    return dt if dt.tzinfo else dt.replace(tzinfo=UTC)


def pass_gate(d: Path, pass_num: int, n_items: int, min_days: float,
              now: datetime | None = None) -> Gate:
    now = now or datetime.now(UTC)
    if pass_num == 1:
        return Gate(True)
    if pass_num == 2:
        p1 = load_labels(d, 1)
        if len({x.item_id for x in p1}) < n_items:
            return Gate(False, f"Pass 2 opens after pass 1 is complete "
                               f"({len(p1)}/{n_items} labeled).")
        available = max(_parse(x.labeled_at) for x in p1) + timedelta(days=min_days)
        if now < available:
            return Gate(False, f"Pass 2 is not open yet. It becomes available on "
                               f"{available.astimezone().strftime('%Y-%m-%d %H:%M')} "
                               f"({min_days:g} days after the last pass-1 label).", available)
        return Gate(True, available_at=available)
    p2 = load_labels(d, 2)
    if len({x.item_id for x in p2}) < n_items:
        return Gate(False, f"Adjudication opens after pass 2 is complete "
                           f"({len(p2)}/{n_items} labeled).")
    return Gate(True)


def disagreements(p1: list[HumanLabel], p2: list[HumanLabel]) -> list[dict[str, Any]]:
    b = {x.item_id: x for x in p2}
    out = []
    for a in p1:
        o = b.get(a.item_id)
        if o is None:
            continue
        fields = [f for f in ("refusal", "harmful") if getattr(a, f) != getattr(o, f)]
        if fields:
            out.append({"item_id": a.item_id, "fields": fields,
                        "pass1": {"refusal": a.refusal, "harmful": a.harmful},
                        "pass2": {"refusal": o.refusal, "harmful": o.harmful}})
    return out


def kappa_report(p1: list[HumanLabel], p2: list[HumanLabel]) -> dict[str, Any]:
    from sklearn.metrics import cohen_kappa_score

    b = {x.item_id: x for x in p2}
    pairs = [(a, b[a.item_id]) for a in p1 if a.item_id in b]

    def kappa(field: str) -> float | None:
        xs = [(str(getattr(a, field)), str(getattr(o, field))) for a, o in pairs
              if getattr(a, field) is not None and getattr(o, field) is not None]
        if not xs:
            return None
        k = cohen_kappa_score([x for x, _ in xs], [y for _, y in xs])
        return None if pd.isna(k) else float(k)

    dis = disagreements(p1, p2)
    return {"n_items": len(pairs), "refusal_kappa": kappa("refusal"),
            "harmful_kappa": kappa("harmful"), "n_disagreements": len(dis),
            "disagreements": dis}


def final_labels(p1: list[HumanLabel], p2: list[HumanLabel], adjudicated: list[HumanLabel]
                 ) -> tuple[list[HumanLabel], list[str]]:
    """Final reference label per item: the agreed label, or the adjudicated one where the
    passes differ. Returns (finals, item ids still awaiting adjudication)."""
    b = {x.item_id: x for x in p2}
    adj = {x.item_id: x for x in adjudicated}
    finals, pending = [], []
    for a in p1:
        o = b.get(a.item_id)
        if o is None:
            pending.append(a.item_id)
        elif (a.refusal, a.harmful) == (o.refusal, o.harmful):
            finals.append(a.model_copy(update={"pass_num": ADJUDICATED_PASS, "seconds": None,
                                               "notes": "agreed"}))
        elif a.item_id in adj:
            finals.append(adj[a.item_id])
        else:
            pending.append(a.item_id)
    return finals, pending


def single_pass_labels(p1: list[HumanLabel], item_ids: list[str]
                       ) -> tuple[list[HumanLabel], list[str]]:
    """Pass-1 labels taken as final (pilot calibration, no test-retest). Returns (finals,
    item ids not yet labeled in pass 1)."""
    finals = [a.model_copy(update={"pass_num": ADJUDICATED_PASS, "seconds": None,
                                   "notes": "single_pass"}) for a in p1]
    labeled = {a.item_id for a in p1}
    return finals, [i for i in item_ids if i not in labeled]


def import_annotations(d: Path, single_pass: bool = False) -> dict[str, Any]:
    """Combine passes into human_labels.parquet and write the reliability report. With
    `single_pass`, pass-1 labels are the final labels and no pass 2 is expected."""
    p1, p2, adj = load_labels(d, 1), load_labels(d, 2), load_labels(d, ADJUDICATED_PASS)
    if single_pass and (p2 or adj):
        raise ValueError(f"{d} has pass-2 or adjudication labels; import it without "
                         "single_pass")
    report = kappa_report(p1, p2)
    if single_pass:
        finals, pending = single_pass_labels(p1, [x["item_id"] for x in load_items(d)])
    else:
        finals, pending = final_labels(p1, p2, adj)
    report.update({"single_pass": single_pass, "n_pass1": len(p1), "n_pass2": len(p2),
                   "n_final": len(finals), "pending_adjudication": pending})
    write_records(d / "human_labels.parquet", [*p1, *p2, *finals])
    (d / "reliability.json").write_text(json.dumps(report, indent=2))
    if single_pass:
        (d / "reliability.md").write_text(
            "# Single-pass labels\n\n- pass-1 labels used as final; no test-retest kappa\n"
            f"- final labels: {len(finals)}; not yet labeled: {len(pending)}\n")
        return report
    lines = ["# Test-retest reliability\n",
             f"- items with both passes: {report['n_items']}",
             f"- Cohen's kappa, refusal (3-class): {report['refusal_kappa']}",
             f"- Cohen's kappa, harmful: {report['harmful_kappa']}",
             f"- disagreements: {report['n_disagreements']}; awaiting adjudication: "
             f"{len(pending)}\n", "| item | fields | pass 1 | pass 2 |", "|---|---|---|---|"]
    lines += [f"| {x['item_id']} | {', '.join(x['fields'])} | {x['pass1']} | {x['pass2']} |"
              for x in report["disagreements"]]
    (d / "reliability.md").write_text("\n".join(lines) + "\n")
    return report
