# 3 分钟系统演示脚本

目标：录一个 3 分钟以内的视频，证明项目不是只做 PPT，而是有真实数据、可运行代码、实验结果和报告产物。

## 演示前准备

打开一个终端，进入项目目录：

```bash
cd /Users/lichenghuang/workspace/homeworks/data-mining/2026-5-30midterm/sentiment-cross-domain
```

推荐使用 Codex 内置 Python，避免系统 Python 缺依赖：

```bash
PY=/Users/lichenghuang/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/bin/python3
```

## 逐秒脚本

### 0:00-0:20 开场

画面：终端 + Finder 或 VS Code 文件树。

口播：

“下面演示我们的跨域情感分析系统。这个系统包含真实处理后的源域和目标域数据、E0-E6 实验矩阵、验证脚本、图表和最终报告。”

### 0:20-0:55 展示项目结构和关键文件

命令：

```bash
ls data/processed results report presentation
```

口播：

“这里可以看到 processed 数据、实验结果、最终报告和答辩材料。正式数据文件是 source_full.csv 和 social_full.csv；结果包括 final_metrics、final_summary 和 error_cases。”

### 0:55-1:35 打印系统摘要

命令：

```bash
$PY scripts/demo_snapshot.py
```

口播：

“这个脚本读取最终产物，不重跑实验，用于快速检查系统状态。可以看到源域 8000 条、目标域 6000 条，其中 unlabeled 目标域 3600 条。实验矩阵包含 E0 到 E6，每个方法 3 个 seed。最佳目标域 F1 是 E3 的 0.927，最平衡 DeltaF1 是 E6 的 -0.019。”

### 1:35-2:10 运行验收脚本

命令：

```bash
$PY scripts/verify_final.py
```

口播：

“接着运行最终验收脚本。它会检查数据规模、目标域 unlabeled、E0-E6 是否完整、每个方法是否 3 个 seed，以及是否有方法满足 DeltaF1 和负面召回要求。”

### 2:10-2:40 展示结果表和失败样本

命令：

```bash
head -8 results/final_summary.csv
head -4 results/final_error_cases.csv
```

口播：

“final_summary.csv 保存了所有方法的均值和标准差；final_error_cases.csv 保存目标域错误案例，用于分析微博文本里的反讽、表情和局部情感词冲突。”

### 2:40-3:00 展示报告和 PPT 产物

命令：

```bash
ls -lh report/final_report.pdf presentation/final_defense_7min.pptx presentation/system_demo_3min_script.md
```

口播：

“最后，报告 PDF、7 分钟答辩 PPT 和 3 分钟演示脚本都已经生成。到这里，数据、代码、实验、报告和答辩材料形成了完整闭环。”

## 如果视频还有 10 秒

可以打开图表或报告：

```bash
open results/final_metrics_chart.png
open report/final_report.pdf
```
