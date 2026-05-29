# 社交媒体情感分析的跨域泛化

数据挖掘课程项目 · 2025 秋

## 团队
- 李乘黄 (3220251475)
- 马啸 (3220251484)
- THAM WAN HEI (3820251057)
- 化润宇 (3220251231)

## 项目简介
基于 SST-2 / IMDB 源域与自建社会事件评论目标域的跨域情感分类,
聚焦数据中心化改进(弱监督伪标注 + 分布偏移诊断)与轻量领域适配。

## 目录结构
- `data/` — 数据(原始数据不入库,只放说明)
- `src/` — 源码
- `experiments/` — 实验脚本与结果
- `report/` — 开题/中期/最终报告

## 中期可复现闭环

当前仓库已补齐一个不依赖 PyTorch / Transformers / scikit-learn 的轻量 smoke benchmark，
用于中期阶段验证数据清洗、源域到目标域迁移、阈值扫描和指标保存链路。

```bash
python run.py experiment=source_only model=tfidf_lr
```

如果本机 Python 尚未安装 `numpy`，请先安装 `requirements.txt` 中的基础依赖，或使用课程/实验室环境。
命令会生成：

- `results/midterm_tfidf_metrics.json`：数据规模、F1、AUC、负面召回率、阈值等指标。
- `results/midterm_tfidf_predictions.csv`：源域和目标域测试集逐样本预测。

中期样例数据位于：

- `data/processed/source_sample.csv`：影评/书评源域样例，train/test 划分。
- `data/processed/social_sample.csv`：社会事件评论目标域样例，train/val/test 划分。

该样例只用于保证工程闭环和报告量化表格可复现；终期实验需要替换为完整 SST-2 / IMDB
与真实爬取的 5k+ 社会事件评论。

## 开题报告
见 `report/proposal.md`
