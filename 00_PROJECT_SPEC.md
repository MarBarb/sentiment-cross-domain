# Project Spec: 跨域情感分析 — KL 领域对齐系统

## 🎯 Objective

构建跨域二分类器 f_θ: X → Y={0,1}，从影评源域迁移到社会事件评论目标域：

1. **源域预训练**: 在大规模标注影评 D_s = {(x_i^s, y_i^s)}_{i=1}^{N_s} 上训练
2. **目标域适应**: 利用少量标注 D_t^L = {(x_j^t, y_j^t)}_{j=1}^{N_t^L} 和大量无标注 D_t^U 适配
3. **KL 特征对齐**: 对齐 P_s(φ(x)) 和 P_t(φ(x))，闭式高斯 KL 散度
4. **评估**: ΔF1 = F1(s) − F1(t)，目标 ΔF1 ≤ 15%，最优阈值 τ 在目标验证集调优

## 📊 任务形式化

源域: D_s = {(x_i^s, y_i^s)}_{i=1}^{N_s}，x ∈ X 为文本，y ∈ Y = {0,1}（负面/正面）
目标域: D_t = {x_j^t}_{j=1}^{N_t}，仅 N_t^L ≪ N_t 的人工标注样本

特征分布: P_s(φ(x))、P_t(φ(x))，φ(·) 为 BERT 编码器

核心损失:
```
L = E_{(x,y)~D_t^L} [ ℓ(f_θ(x), y) ] + λ · KL( P_s(φ(x)) ‖ P_t(φ(x)) )
```

## 📦 数据集

| 数据集 | 用途 | 规模 | 特征 |
|--------|------|------|------|
| SST-2 | 源域训练 | ~67k 样本 | 短文本(~10词), 均衡二分类 |
| IMDB | 源域训练/跨域测试 | 50k 样本 | 长文本(~230词), 均衡二分类 |
| 社会事件评论 | 目标域 | 预计 5k+ 有效 | 中等长度(~40词), 负面偏重(约2:1~4:1) |

## 🧮 技术栈

- **Framework**: PyTorch 2.3+, HuggingFace Transformers
- **Config**: Hydra (YAML-based)
- **Tracking**: Weights & Biases
- **PEFT**: LoRA / Adapter (参数高效微调)
- **弱监督**: Snorkel (伪标注)
- **Hardware**: Single GPU (≥16GB VRAM); DDP-ready

## 🗂 模块契约

| 模块 | 输入 | 输出 | 负责 Agent |
|------|------|------|-----------|
| Data | 原始文件 | `DataLoader(s, t)` | Agent-1 |
| Model | config | `nn.Module` (φ encoder + classifier head) | Agent-2 |
| Trainer | model + loaders | trained checkpoint | Agent-3 |
| Evaluator | checkpoint | metrics dict {F1_s, F1_t, ΔF1, τ*} | Agent-4 |
| Runner | all configs | experiment log + W&B dashboard | Agent-5 |

## 🧪 消融实验矩阵

| ID | 模型 | 源域训练 | 弱监督伪标 | 领域适配层 | 对抗域判别 | 验证点 |
|----|------|---------|-----------|-----------|-----------|--------|
| E0 | TF-IDF+LR | ✓ | × | × | × | 浅层基线下限 |
| E1 | BERT | ✓ | × | × | × | 跨域衰减基准 |
| E2 | BERT | ✓ | ✓ | × | × | 弱监督独立贡献 |
| E3 | BERT | ✓ | × | ✓ | × | 适配器独立贡献 |
| E4 | BERT | ✓ | ✓ | ✓ | × | 两者组合 |
| E5 | BERT | ✓ | ✓ | ✓ | ✓ | 完整方案 |
| E6 | RoBERTa | ✓ | ✓ | ✓ | ✓ | backbone 替换 |

核心对照: E1 vs E3 (适配器), E1 vs E2 (弱监督), E4 vs E5 (对抗训练)

## 📏 评估指标

### 技术指标
- Macro-F1 / Weighted-F1 (主指标, 应对类别不平衡)
- AUC-ROC (阈值无关)
- ΔF1 = F1(s) − F1(t) (核心指标, 目标 ≤ 15%)
- KL / MMD 分布距离 (数据层诊断)

### 业务指标
- 目标域负面情感 Recall ≥ 85%
- 人工抽样一致性 Cohen's Kappa ≥ 0.7

## ✅ Acceptance Criteria

- [ ] `python run.py experiment=baseline` 可复现
- [ ] ΔF1 ≤ 15% on 目标域
- [ ] 各模块单元测试通过
- [ ] W&B dashboard 自动记录 loss curves + ΔF1 + τ
- [ ] 消融实验 E0-E6 全部可运行
- [ ] 三随机种子求均值±标准差

## 📅 排期 (16周)

| 周次 | 里程碑 |
|------|--------|
| W1-2 | 爬虫 pipeline、数据清洗、首批 5k 目标域样本入库 |
| W3 | 数据审计报告 (KL/MMD、分布可视化、类别分布) |
| W4-5 | B0、B1 基线跑通，建立跨域衰减基准 |
| W6-8 | 领域适配层 + 弱监督伪标注实现 (E2, E3, E4) |
| W9-11 | 对抗训练 E5、RoBERTa 对比 E6、完整消融 |
| W13-15 | 结果复查、补充实验、业务指标人工评测、报告撰写 |
| W16 | 最终答辩 |
