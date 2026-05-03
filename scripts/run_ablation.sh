#!/bin/bash
# 消融实验批量运行脚本
# 用法: bash scripts/run_ablation.sh

set -e

SEEDS=(42 123 456)

echo "=== E0: TF-IDF + LR Baseline ==="
python run.py experiment=source_only model=tfidf_lr

echo "=== E1: BERT zero-shot ==="
python run.py experiment=source_only

for seed in "${SEEDS[@]}"; do
    echo "=== E2: +WeakSup (seed=$seed) ==="
    python run.py experiment=weak_sup_only seed=$seed

    echo "=== E3: +Adapter (seed=$seed) ==="
    python run.py experiment=finetune_only seed=$seed

    echo "=== E4: +Both (seed=$seed) ==="
    python run.py experiment=kl_align seed=$seed

    echo "=== E5: Full Method (seed=$seed) ==="
    python run.py experiment=full_method seed=$seed

    echo "=== E6: RoBERTa (seed=$seed) ==="
    python run.py experiment=roberta_full seed=$seed
done

echo "=== All ablation experiments complete ==="
