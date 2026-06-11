# 社交媒体情感分析的跨域泛化

数据挖掘课程项目 · 最终交付版

## 团队

- 李乘黄 (3220251475)
- 马啸 (3220251484)
- THAM WAN HEI (3820251057)
- 化润宇 (3220251231)

## 项目简介

本项目研究评论源域到微博社交媒体目标域的跨域情感分类。最终版本已从中期
sample smoke benchmark 升级为真实公开语料上的可复现实验矩阵：

- 源域：ChineseNlpCorpus `waimai_10k` 外卖评论。
- 目标域：HuggingFace `dirtycomputer/weibo_senti_100k` 微博情感数据。
- 目标域构造为负:正=2:1，包含少量标注集与 3600 条 unlabeled 样本。
- 实验覆盖 E0-E6、3 个随机种子、Macro-F1 / Weighted-F1 / AUC / DeltaF1 / KL / MMD / 负面召回率。

仓库中保留 BERT/RoBERTa、KL 对齐和训练器骨架；最终可复现实验采用 CPU 友好的
lexical/hash 特征实现，保证在无 GPU、无外部服务账号的课程环境中也能一键复现完整消融。

## 快速复现

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
./scripts/run_final.sh
python scripts/verify_final.py
```

如果使用 Codex 桌面环境的内置 Python：

```bash
PYTHON=/Users/lichenghuang/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/bin/python3 ./scripts/run_final.sh
PYTHON=/Users/lichenghuang/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/bin/python3 scripts/verify_final.py
```

兼容项目规范中的 baseline 命令：

```bash
python run.py experiment=baseline
```

## 数据

正式数据文件：

- `data/processed/source_full.csv`：8000 条源域评论，train/val/test=5600/1200/1200。
- `data/processed/social_full.csv`：6000 条目标域微博，train/val/test/unlabeled=600/600/1200/3600。
- `results/final_data_audit.json`：原始行数、采样规模、类别比例、平均长度和下载来源。

原始公开 CSV 会由 `scripts/prepare_real_data.py` 下载到 `data/raw/`，该目录下的大文件已被
`.gitignore` 排除；processed 数据和审计文件纳入版本管理，便于直接检查与复现。

中期样例数据仍保留：

- `data/processed/source_sample.csv`
- `data/processed/social_sample.csv`

它们只用于早期 smoke benchmark，不是最终实验默认数据。

## 实验矩阵

| ID | 方法 | 作用 |
| :--- | :--- | :--- |
| E0 | TF-IDF+LR source-only | 浅层源域 baseline |
| E1 | Source-only strong lexical encoder | 增强 lexical source-only |
| E2 | Weak supervision pseudo labels | 使用 unlabeled 目标域伪标注 |
| E3 | Target adapter calibration | 少量目标域金标校准 |
| E4 | Weak supervision + adapter | 伪标注与目标域校准组合 |
| E5 | Domain-adversarial feature filtering | 过滤强领域特异 n-gram |
| E6 | Backbone replacement char1-5 | 替换为更宽 char1-5 特征空间 |

核心入口：

```bash
python run.py experiment=final
python scripts/generate_final_report.py
python scripts/verify_final.py
```

## 最终结果

3 个随机种子均值摘要：

| ID | F1(source) | F1(target) | DeltaF1 | AUC(target) | Neg Recall |
| :--- | ---: | ---: | ---: | ---: | ---: |
| E0 | 0.848 | 0.625 | 0.223 | 0.685 | 0.718 |
| E1 | 0.866 | 0.644 | 0.222 | 0.719 | 0.873 |
| E2 | 0.852 | 0.753 | 0.099 | 0.834 | 0.879 |
| E3 | 0.860 | 0.927 | -0.068 | 0.980 | 0.958 |
| E4 | 0.848 | 0.897 | -0.049 | 0.963 | 0.913 |
| E5 | 0.851 | 0.681 | 0.170 | 0.775 | 0.898 |
| E6 | 0.871 | 0.890 | -0.019 | 0.962 | 0.935 |

E3 的目标域 Macro-F1 最佳；E6 的 DeltaF1 最平衡。至少一个方法满足
`|DeltaF1| <= 0.15` 且目标域负面召回率 `>= 0.85`。

## 交付物

- `report/final_report.md`
- `report/final_report.pdf`
- `results/final_metrics.json`
- `results/final_summary.csv`
- `results/final_runs.csv`
- `results/final_error_cases.csv`
- `results/final_metrics_chart.png`

## 答辩材料

- `presentation/final_defense_7min.pptx`：7 分钟最终汇报 PPT。
- `presentation/speaker_notes_7min.md`：逐页讲稿。
- `presentation/system_demo_3min_script.md`：3 分钟系统演示脚本。
- `presentation/recording_guide.md`：录屏和拍摄指南。

中期材料仍保留在 `report/midterm.md` / `report/midterm.pdf`，开题材料见 `report/proposal.md`。
