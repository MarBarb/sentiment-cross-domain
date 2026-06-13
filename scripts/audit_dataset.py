"""Run final processed dataset audit checks."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.evaluation.dataset_audit import audit_cross_domain_dataset, write_audit_report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Audit final cross-domain processed CSV files.")
    parser.add_argument("--source", default="data/processed/source_full.csv")
    parser.add_argument("--target", default="data/processed/social_full.csv")
    parser.add_argument("--output", default="results/dataset_audit.json")
    parser.add_argument("--no-write", action="store_true", help="Print the report without writing JSON.")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    report = audit_cross_domain_dataset(source_path=args.source, target_path=args.target)
    text = json.dumps(report, ensure_ascii=False, indent=2)
    print(text)

    if not args.no_write:
        write_audit_report(report, args.output)

    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
