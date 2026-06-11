# 数据挖掘课程最终项目报告

**题目**：社交媒体情感分析的跨域泛化：面向分布偏移的 Data-Centric 改进

**团队成员**：李乘黄、马啸、THAM WAN HEI、化润宇

**仓库链接**：https://github.com/MarBarb/sentiment-cross-domain

**最终交付日期**：2026-06-03

## 摘要

本项目研究评论源域到微博社交媒体目标域的二分类情感迁移问题。我们使用真实公开语料构造源域 8000 条、目标域 6000 条数据，并围绕弱监督伪标注、少量目标域校准、领域特征过滤和 backbone 替换设计 E0-E6 消融实验。结果显示，E3 Target adapter calibration 在目标域取得最高 Macro-F1=0.927，E6 Backbone replacement char1-5 的 DeltaF1=-0.019 最平衡。失败分析表明，微博中的转发链、表情、反讽和局部情感词冲突仍是浅层特征方法的主要误差来源。

**关键词**：跨域情感分析；领域偏移；弱监督；消融实验；Macro-F1

## 1. 项目基本信息

- **项目名称**：社交媒体情感分析的跨域泛化：面向分布偏移的 Data-Centric 改进
- **仓库链接**：https://github.com/MarBarb/sentiment-cross-domain
- **团队成员**：李乘黄、马啸、THAM WAN HEI、化润宇
- **最终交付日期**：2026-06-03
- **复现命令**：`./scripts/run_final.sh && python scripts/verify_final.py`

## 2. 问题定义与目标

项目目标是从评论源域迁移到微博社交媒体目标域，构建二分类情感模型。重点不是单一模型堆叠，而是验证数据层诊断、弱监督伪标注、少量目标域适配、领域特征过滤和特征 backbone 替换对跨域性能衰减的影响。

验收指标包括 Macro-F1、Weighted-F1、AUC、DeltaF1、目标域负面召回率，以及特征层 KL/MMD 诊断。

## 3. 数据来源与审计

数据来自公开语料，可由 `scripts/prepare_real_data.py` 重新生成：

- 源域：ChineseNlpCorpus `waimai_10k`，下载地址：https://raw.githubusercontent.com/SophonPlus/ChineseNlpCorpus/master/datasets/waimai_10k/waimai_10k.csv
- 目标域：HuggingFace `dirtycomputer/weibo_senti_100k`，下载地址：https://huggingface.co/datasets/dirtycomputer/weibo_senti_100k/resolve/main/weibo_senti_100k.csv

**表 1 数据集规模与划分**

| 域 | raw rows | processed rows | train | val | test | unlabeled | 正例比例 | 平均长度 |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 源域 waimai review | 11987 | 8000 | 5600 | 1200 | 1200 | 0 | 50.0% | 22.9 |
| 目标域 weibo social | 119000 | 6000 | 600 | 600 | 1200 | 3600 | 33.3% | 51.9 |

目标域按负:正=2:1 构造，模拟社会事件评论中负面/质疑声音偏重的场景；仅 10% 作为有标注训练集，60% 作为未标注数据用于弱监督伪标注。

## 4. 方法与消融矩阵

最终实验覆盖 **E0-E6** 七个方法，并对每个方法运行 3 个随机种子。

**表 2 方法与消融设置**

| ID | 方法 | 说明 |
| :--- | :--- | :--- |
| E0 | TF-IDF+LR source-only | 浅层 TF-IDF+LR，仅源域训练。 |
| E1 | Source-only strong lexical encoder | 增强 lexical encoder，仅源域训练，加入情感词典特征。 |
| E2 | Weak supervision pseudo labels | 弱监督伪标注，从目标域 unlabeled 中选高置信样本加入训练。 |
| E3 | Target adapter calibration | 目标域 adapter calibration，使用少量目标域金标样本进行适配。 |
| E4 | Weak supervision + adapter | 弱监督 + adapter 组合。 |
| E5 | Domain-adversarial feature filtering | 领域特征过滤，模拟 domain-adversarial 去除强域特异 n-gram。 |
| E6 | Backbone replacement char1-5 | backbone 替换，使用 char1-5 更宽特征空间。 |

## 5. 最终实验结果

所有结果均为 3 个随机种子（42/123/456）的均值与标准差。

**表 3 最终实验指标汇总**

| ID | F1(source) | F1(target) | DeltaF1 | AUC(target) | Neg Recall | Pos Recall | Weighted-F1 |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| E0 | 0.848±0.001 | 0.625±0.003 | 0.223±0.002 | 0.685 | 0.718 | 0.541 | 0.663 |
| E1 | 0.866±0.003 | 0.644±0.003 | 0.222±0.001 | 0.719 | 0.873 | 0.402 | 0.698 |
| E2 | 0.852±0.006 | 0.753±0.006 | 0.099±0.012 | 0.834 | 0.879 | 0.609 | 0.784 |
| E3 | 0.860±0.001 | 0.927±0.001 | -0.068±0.002 | 0.980 | 0.958 | 0.892 | 0.936 |
| E4 | 0.848±0.001 | 0.897±0.007 | -0.049±0.007 | 0.963 | 0.913 | 0.894 | 0.908 |
| E5 | 0.851±0.001 | 0.681±0.005 | 0.170±0.005 | 0.775 | 0.898 | 0.443 | 0.729 |
| E6 | 0.871±0.015 | 0.890±0.005 | -0.019±0.020 | 0.962 | 0.935 | 0.840 | 0.903 |

目标域 F1 最佳方法为 **E3 Target adapter calibration**，F1(target)=0.927，负面召回=0.958。
DeltaF1 最平衡方法为 **E6 Backbone replacement char1-5**，DeltaF1=-0.019。

结论：少量目标域金标校准（E3）是收益最大的模块；弱监督伪标注（E2）能将目标域 F1 从 E1 的 0.644 提升到 0.753；E5 虽然降低了 MMD，但过度过滤导致正例召回下降，是一个负向消融。

![图 1 最终实验结果对比](../results/final_metrics_chart.png)

## 6. 失败案例与局限分析

错误样本保存在 `results/final_error_cases.csv`。主要失败模式包括：

- 微博文本中大量转发链、表情和反讽表达，线性 n-gram 模型容易把局部正向词误判为整体正向。
- 目标域中“祝福/哈哈/鼓掌”等词在不同语境下可能同时出现在正负样本中，造成词面冲突。
- 领域特征过滤能够降低 MMD，但如果过滤过强，会删除目标域关键情感触发词。

**表 4 失败案例摘录**

| 方法 | 真实标签 | 预测 | 正类概率 | 文本摘录 |
| :--- | ---: | ---: | ---: | :--- |
| E0 | 0 | 1 | 0.503 | 简体字毕竟写和认较为方便，不能一概否认。//：不该丢的都丢了~[衰] //：。 |
| E0 | 1 | 0 | 0.340 | [飞机]捧场去??? //:?庄 是俺的第10000个粉丝，支持一下阳江程村生蚝！[鼓掌] |

局限性：当前最终可复现实验采用 CPU 友好的 lexical/hash 特征实现，能够稳定复现消融结论，但对跨句语义、反讽和表情组合的建模能力弱于完整 BERT/RoBERTa 深度微调。后续可在保留同一数据 split 的前提下加入上下文编码器，并用困难样本主动学习验证改进。

## 7. 复现与交付

核心命令：

```bash
./scripts/run_final.sh
python scripts/verify_final.py
```

核心产物：

- `data/processed/source_full.csv`
- `data/processed/social_full.csv`
- `results/final_metrics.json`
- `results/final_summary.csv`
- `results/final_error_cases.csv`
- `results/final_metrics_chart.png`
- `report/final_report.md` / `report/final_report.pdf`

## 8. AI 工具使用声明

项目使用 ChatGPT/Codex 辅助代码补全、实验脚本整理、报告生成与调试；数据处理、指标解释和最终结论均以本地可复现运行结果为准。
