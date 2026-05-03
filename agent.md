# 🏗️ 跨域情感分析：KL 领域对齐系统

社交媒体情感分析的跨域泛化 — 面向分布偏移的 Data-Centric 改进

## 🎯 项目目标

从影评源域 (SST-2/IMDB) 迁移到社会事件评论目标域，通过 KL 散度对齐特征分布，
最小化跨域性能衰减 ΔF1 ≤ 15%。

## 📦 技术栈

- **Framework**: PyTorch 2.3+ / HuggingFace Transformers
- **Config**: Hydra (YAML-based)
- **Tracking**: Weights & Biases
- **PEFT**: LoRA / Adapter
- **弱监督**: Snorkel

## 🗂 项目结构

```
.
├── 00_PROJECT_SPEC.md          # 总纲 (给主控 Agent)
├── 01_DATA_PIPELINE.md         # 数据 Agent
├── 02_MODEL_ARCHITECTURE.md    # 模型 Agent
├── 03_TRAINING_LOOP.md         # 训练 Agent (核心: KL 对齐)
├── 04_EVALUATION.md            # 评估 Agent (ΔF1 + τ 阈值)
├── 05_EXPERIMENT_RUNNER.md     # 实验编排 Agent
├── run.py                      # Hydra 入口
├── configs/                    # YAML 配置
│   ├── config.yaml
│   ├── data/
│   ├── model/
│   ├── train/
│   ├── experiment/
│   └── sweeps/
├── src/
│   ├── data/                   # 数据管道
│   │   ├── datamodule.py
│   │   ├── datasets.py
│   │   └── cleaner.py
│   ├── models/                 # 模型架构
│   │   ├── model.py
│   │   ├── domain_discriminator.py
│   │   └── layers.py
│   ├── training/               # 训练循环
│   │   ├── trainer.py
│   │   ├── kl_loss.py
│   │   ├── ema.py
│   │   └── losses.py
│   ├── evaluation/             # 评估
│   │   ├── metrics.py
│   │   ├── evaluator.py
│   │   └── visualize.py
│   └── utils/
├── tests/                      # 单元测试
├── data/
│   ├── raw/
│   └── processed/
├── scripts/                    # 运行脚本
└── report/                     # 论文/报告
```

## 🚀 快速开始

```bash
# 1. 安装依赖
pip install -r requirements.txt

# 2. 运行测试
pytest tests/ -v

# 3. 运行基线实验
python run.py experiment=source_only

# 4. 运行完整方案 (E5)
python run.py experiment=full_method train.lambda_kl=0.1

# 5. 运行全部消融实验
bash scripts/run_ablation.sh
```

## 🧪 消融实验

| ID | 模型 | 源域训练 | 弱监督伪标 | 领域适配层 | 对抗域判别 | 验证点 |
|----|------|---------|-----------|-----------|-----------|--------|
| E0 | TF-IDF+LR | ✓ | × | × | × | 浅层基线下限 |
| E1 | BERT | ✓ | × | × | × | 跨域衰减基准 |
| E2 | BERT | ✓ | ✓ | × | × | 弱监督独立贡献 |
| E3 | BERT | ✓ | × | ✓ | × | 适配器独立贡献 |
| E4 | BERT | ✓ | ✓ | ✓ | × | 两者组合 |
| E5 | BERT | ✓ | ✓ | ✓ | ✓ | 完整方案 |
| E6 | RoBERTa | ✓ | ✓ | ✓ | ✓ | backbone 替换 |

## 📏 核心指标

- **ΔF1** = F1(source) − F1(target) ≤ 15%
- **负面 Recall** ≥ 85%
- **τ** 在目标域验证集上调优

## 📅 排期 (16周)

| 周次 | 里程碑 |
|------|--------|
| W1-2 | 数据 pipeline + 首批 5k 目标域样本 |
| W3 | 数据审计报告 |
| W4-5 | 基线 B0/B1 跑通 |
| W6-8 | 领域适配 + 弱监督 (E2-E4) |
| W9-11 | 对抗训练 + 完整消融 (E5-E6) |
| W13-15 | 补充实验 + 报告 |
| W16 | 答辩 |

## 🤖 Agent 协作模式

按 DAG 顺序执行:
```
        ┌── Data-Agent ──┐
Spec ──►│                ├──► Trainer-Agent ──┐
        └── Model-Agent ─┘                    ├──► Runner-Agent
                          └─► Eval-Agent  ────┘
```

Data 与 Model 可**并行**开发，Trainer 依赖两者就绪。
