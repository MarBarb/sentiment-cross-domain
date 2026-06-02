"""实验入口.

支持三个无需 GPU 的可复现实验：

    python run.py experiment=source_only model=tfidf_lr
    python run.py experiment=baseline
    python run.py experiment=final

保留 `key=value` 风格参数，方便之后平滑迁回 Hydra。
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

from src.baselines import run_tfidf_experiment
from src.experiments import run_final_experiments

logger = logging.getLogger(__name__)


def _parse_kv_args(argv):
    parsed = {}
    positional = []
    for arg in argv:
        if "=" in arg and not arg.startswith("--"):
            key, value = arg.split("=", 1)
            parsed[key.strip()] = value.strip()
        else:
            positional.append(arg)
    return parsed, positional


def build_parser():
    parser = argparse.ArgumentParser(description="跨域情感分析实验入口")
    parser.add_argument("--model", default="tfidf_lr", help="当前可运行: tfidf_lr")
    parser.add_argument("--experiment", default="source_only", help="source_only、baseline 或 final")
    parser.add_argument("--source-path", default=None)
    parser.add_argument("--target-path", default=None)
    parser.add_argument("--output-dir", default="results")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-features", type=int, default=3000)
    parser.add_argument("--epochs", type=int, default=900)
    return parser


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    kv_args, remaining = _parse_kv_args(argv)
    parser = build_parser()
    args = parser.parse_args(remaining)

    # 兼容原命令：python run.py experiment=source_only model=tfidf_lr
    if "model" in kv_args:
        args.model = kv_args["model"]
    if "experiment" in kv_args:
        args.experiment = kv_args["experiment"]
    if "seed" in kv_args:
        args.seed = int(kv_args["seed"])

    logging.basicConfig(level=logging.INFO, format="%(levelname)s:%(name)s:%(message)s")

    if args.experiment == "final":
        source_path = args.source_path or "data/processed/source_full.csv"
        target_path = args.target_path or "data/processed/social_full.csv"
        summary, metrics_path = run_final_experiments(
            source_path=source_path,
            target_path=target_path,
            output_dir=args.output_dir,
        )
        logger.info("Final experiment matrix complete.")
        logger.info("Metrics saved to %s", metrics_path)
        print(json.dumps(summary["summary"], ensure_ascii=False, indent=2))
        return

    if args.experiment == "baseline":
        source_path = args.source_path or "data/processed/source_full.csv"
        target_path = args.target_path or "data/processed/social_full.csv"
        summary, metrics_path, pred_path = run_tfidf_experiment(
            source_path=source_path,
            target_path=target_path,
            output_dir=args.output_dir,
            seed=args.seed,
            max_features=args.max_features,
            epochs=args.epochs,
            artifact_prefix="baseline_tfidf",
        )
        logger.info("Baseline experiment complete.")
        logger.info("Metrics saved to %s", metrics_path)
        logger.info("Predictions saved to %s", pred_path)
        print(json.dumps(summary["experiments"], ensure_ascii=False, indent=2))
        return

    if args.model != "tfidf_lr":
        raise SystemExit(
            f"当前轻量入口只支持 model=tfidf_lr，收到 model={args.model}。"
            "BERT/RoBERTa 训练需安装 requirements.txt 并接通 datamodule。"
        )

    source_path = args.source_path or "data/processed/source_sample.csv"
    target_path = args.target_path or "data/processed/social_sample.csv"
    summary, metrics_path, pred_path = run_tfidf_experiment(
        source_path=source_path,
        target_path=target_path,
        output_dir=args.output_dir,
        seed=args.seed,
        max_features=args.max_features,
        epochs=args.epochs,
    )

    logger.info("Experiment %s complete.", args.experiment)
    logger.info("Metrics saved to %s", metrics_path)
    logger.info("Predictions saved to %s", pred_path)

    print(json.dumps(summary["experiments"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
