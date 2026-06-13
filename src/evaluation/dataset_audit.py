"""Dataset audit helpers for the final cross-domain handoff.

The final experiment keeps processed CSV files in the repository so that the
project can be inspected without downloading raw corpora. These helpers provide
small, dependency-light checks around the handoff data contract:

- required columns are present
- required splits exist
- each evaluated split has both labels
- target data has enough rows and enough unlabeled examples
- unlabeled rows are tracked as an audit note, not as training labels
"""
from __future__ import annotations

import csv
import json
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


REQUIRED_COLUMNS = {"text", "label", "split"}
SOURCE_SPLITS = ("train", "val", "test")
TARGET_SPLITS = ("train", "val", "test", "unlabeled")
EVALUATED_SPLITS = ("train", "val", "test")


@dataclass(frozen=True)
class SplitAudit:
    name: str
    rows: int
    positive: int
    negative: int
    missing_label: int
    invalid_label: int
    avg_chars: float

    @property
    def positive_rate(self) -> float:
        labeled = self.positive + self.negative
        return self.positive / labeled if labeled else 0.0

    def to_dict(self) -> dict:
        return {
            "rows": self.rows,
            "positive": self.positive,
            "negative": self.negative,
            "missing_label": self.missing_label,
            "invalid_label": self.invalid_label,
            "positive_rate": self.positive_rate,
            "avg_chars": self.avg_chars,
        }


def _parse_label(value: str | None) -> int | None:
    if value is None or value.strip() == "":
        return None
    try:
        return int(value)
    except ValueError:
        return None


def read_processed_rows(path: str | Path) -> tuple[list[dict], list[str]]:
    """Read a processed CSV and return rows plus schema-level issues."""
    csv_path = Path(path)
    issues: list[str] = []
    if not csv_path.exists():
        return [], [f"{csv_path} does not exist"]

    rows: list[dict] = []
    with csv_path.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        fieldnames = set(reader.fieldnames or [])
        missing = REQUIRED_COLUMNS - fieldnames
        if missing:
            return [], [f"{csv_path} missing columns: {sorted(missing)}"]

        for line_no, row in enumerate(reader, start=2):
            label = _parse_label(row.get("label"))
            rows.append(
                {
                    "line_no": line_no,
                    "text": (row.get("text") or "").strip(),
                    "label": label,
                    "raw_label": row.get("label", ""),
                    "split": (row.get("split") or "").strip().lower(),
                    "domain": (row.get("domain") or csv_path.stem).strip(),
                }
            )
    return rows, issues


def summarize_split(name: str, rows: Iterable[dict]) -> SplitAudit:
    values = list(rows)
    labels = Counter(row["label"] for row in values)
    invalid = sum(
        1 for row in values
        if str(row["raw_label"]).strip() not in {"", "0", "1"} and row["label"] not in (0, 1)
    )
    lengths = [len(row["text"]) for row in values]
    return SplitAudit(
        name=name,
        rows=len(values),
        positive=labels[1],
        negative=labels[0],
        missing_label=labels[None],
        invalid_label=invalid,
        avg_chars=sum(lengths) / len(lengths) if lengths else 0.0,
    )


def summarize_file(path: str | Path, required_splits: Iterable[str]) -> dict:
    rows, issues = read_processed_rows(path)
    split_names = sorted({row["split"] for row in rows if row["split"]})
    required = tuple(required_splits)
    split_audits = {
        split: summarize_split(split, (row for row in rows if row["split"] == split)).to_dict()
        for split in split_names
    }

    for split in required:
        if split not in split_audits:
            issues.append(f"{Path(path).name} missing split: {split}")

    blank_text = sum(1 for row in rows if not row["text"])
    if blank_text:
        issues.append(f"{Path(path).name} has blank text rows: {blank_text}")

    duplicate_texts = len(rows) - len({(row["text"], row["split"], row["domain"]) for row in rows})
    label_issues = sum(1 for row in rows if row["split"] in EVALUATED_SPLITS and row["label"] not in (0, 1))
    if label_issues:
        issues.append(f"{Path(path).name} has invalid evaluated labels: {label_issues}")

    for split in EVALUATED_SPLITS:
        audit = split_audits.get(split)
        if audit and (audit["positive"] == 0 or audit["negative"] == 0):
            issues.append(f"{Path(path).name}:{split} must contain both labels")

    return {
        "path": str(path),
        "rows": len(rows),
        "domains": sorted({row["domain"] for row in rows if row["domain"]}),
        "splits": split_audits,
        "duplicate_text_split_domain_rows": duplicate_texts,
        "issues": issues,
    }


def _check(condition: bool, message: str, checks: list[dict]) -> None:
    checks.append({"ok": bool(condition), "message": message})


def audit_cross_domain_dataset(
    source_path: str | Path = "data/processed/source_full.csv",
    target_path: str | Path = "data/processed/social_full.csv",
    min_source_rows: int = 8000,
    min_target_rows: int = 5000,
    min_target_unlabeled: int = 3000,
) -> dict:
    """Audit the final source/target processed CSV files."""
    source = summarize_file(source_path, SOURCE_SPLITS)
    target = summarize_file(target_path, TARGET_SPLITS)
    checks: list[dict] = []

    _check(source["rows"] >= min_source_rows, f"source rows >= {min_source_rows}", checks)
    _check(target["rows"] >= min_target_rows, f"target rows >= {min_target_rows}", checks)

    for split in SOURCE_SPLITS:
        _check(split in source["splits"], f"source split present: {split}", checks)
    for split in TARGET_SPLITS:
        _check(split in target["splits"], f"target split present: {split}", checks)

    target_unlabeled = target["splits"].get("unlabeled", {})
    _check(
        target_unlabeled.get("rows", 0) >= min_target_unlabeled,
        f"target unlabeled rows >= {min_target_unlabeled}",
        checks,
    )

    target_train = target["splits"].get("train", {})
    rate = float(target_train.get("positive_rate", 0.0))
    _check(0.20 <= rate <= 0.50, "target train positive rate in expected imbalanced range", checks)

    issues = [*source["issues"], *target["issues"]]
    _check(not issues, "no schema or split issues", checks)

    unlabeled_visible_labels = target_unlabeled.get("positive", 0) + target_unlabeled.get("negative", 0)
    notes = []
    if unlabeled_visible_labels:
        notes.append(
            "target unlabeled rows keep public labels for audit reproducibility; "
            "training code must use unlabeled text only"
        )

    passed = all(check["ok"] for check in checks)
    return {
        "passed": passed,
        "source": source,
        "target": target,
        "checks": checks,
        "notes": notes,
    }


def write_audit_report(report: dict, path: str | Path) -> Path:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return output
