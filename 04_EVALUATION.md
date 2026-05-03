# Agent-4: Evaluation — ΔF1 & 阈值 τ & 消融报表

## Task

实现 `src/evaluation/metrics.py` 和 `src/evaluation/evaluator.py`。

## 核心定义

- `F1(s)`, `F1(t)`: 源域/目标域测试集上的 F1 分数
- `ΔF1 = F1(s) − F1(t)`: 跨域泛化差距，**越小越好**
- `τ`: sigmoid(logit) 上的决策阈值，在**目标域验证集**上扫描 [0.05, 0.95, step=0.01] 调优

## 为什么在目标域上调 τ?

因为 P_t 可能在 KL 对齐后仍偏移最优操作点。避免源域阈值偏置，使 ΔF1 真实反映**特征层面**的域差距。

## API

```python
def evaluate(model, loader_s_test, loader_t_val, loader_t_test) -> dict:
    """
    完整评估流程:
    1. 在目标域验证集上找 τ* = argmax_τ F1_t_val(τ)
    2. 用 τ* 计算 F1_s_test, F1_t_test
    3. 计算 AUC-ROC, KL/MMD 分布距离
    4. 返回 metrics dict
    """
    return {
        "F1_s": ...,
        "F1_t": ...,
        "deltaF1": ...,
        "tau": ...,
        "AUC_s": ...,
        "AUC_t": ...,
        "KL_divergence": ...,
        "MMD": ...,
    }
```

## 评估指标完整列表

### 技术指标

| 指标 | 公式 | 目标 | 说明 |
|------|------|------|------|
| Macro-F1 | (F1_pos + F1_neg) / 2 | 最大化 | 主指标, 应对类别不平衡 |
| Weighted-F1 | Σ c_i · F1_c | 最大化 | 按类别频率加权 |
| AUC-ROC | — | 最大化 | 阈值无关 |
| ΔF1 | F1(s) − F1(t) | ≤ 15% | 核心指标 |
| KL_div | KL(P_s(φ) ‖ P_t(φ)) | 最小化 | 特征分布距离 |
| MMD² | maximum mean discrepancy | 最小化 | 替代分布距离度量 |

### 业务指标

| 指标 | 目标 | 说明 |
|------|------|------|
| 负面 Recall | ≥ 85% | 漏报负面对舆情业务代价最大 |
| Cohen's Kappa | ≥ 0.7 | 人工标注一致性 |

## 阈值扫描

```python
def find_optimal_threshold(probs, labels, metric='f1'):
    """扫描 [0.05, 0.95], step=0.01, 找最优阈值"""
    best_tau, best_score = 0.5, 0.0
    for tau in np.arange(0.05, 0.95, 0.01):
        preds = (probs >= tau).astype(int)
        score = f1_score(labels, preds, average='macro')
        if score > best_score:
            best_tau, best_score = tau, score
    return best_tau, best_score
```

## 分布距离计算

```python
def compute_kl_features(feats_s, feats_t):
    """计算特征空间的 KL 散度 (高斯近似)"""
    mu_s, var_s = feats_s.mean(0), feats_s.var(0) + 1e-8
    mu_t, var_t = feats_t.mean(0), feats_t.var(0) + 1e-8
    return gaussian_kl(mu_s, var_s, mu_t, var_t)

def compute_mmd(feats_s, feats_t, kernel='rbf'):
    """MMD² with RBF kernel"""
    ...
```

## 消融报表 (自动生成)

```
| Method         | F1(s) | F1(t) | ΔF1  | τ*   | AUC(s) | AUC(t) |
|----------------|-------|-------|------|------|--------|--------|
| E0: TF-IDF+LR  | 0.78  | 0.55  | 0.23 | 0.50 | 0.82   | 0.60   |
| E1: BERT       | 0.91  | 0.62  | 0.29 | 0.50 | 0.95   | 0.68   |
| E2: +WeakSup   | 0.90  | 0.72  | 0.18 | 0.45 | 0.94   | 0.78   |
| E3: +Adapter   | 0.89  | 0.75  | 0.14 | 0.43 | 0.93   | 0.81   |
| E4: +Both      | 0.89  | 0.80  | 0.09 | 0.44 | 0.93   | 0.85   |
| E5: +DANN      | 0.88  | 0.83  | 0.05 | 0.47 | 0.92   | 0.88   |
| E6: RoBERTa    | 0.90  | 0.85  | 0.05 | 0.46 | 0.94   | 0.89   |
```

## 可视化

```python
def plot_tsne(feats_s, feats_t, save_path):
    """t-SNE 可视化 φ(x), 源域 vs 目标域特征分布"""

def plot_threshold_scan(probs, labels, save_path):
    """τ vs F1 曲线"""

def plot_loss_curves(log_path, save_path):
    """训练损失曲线 (L_sup, L_kl, L_total)"""
```

## Tests (`tests/test_evaluation.py`)

- [ ] ΔF1 符号正确 (F1_s ≥ F1_t 时为正)
- [ ] τ 搜索返回 argmax
- [ ] 当 probs == labels 时 F1 == 1.0
- [ ] KL 在相同分布时 == 0
- [ ] MMD 在相同分布时 ≈ 0
- [ ] 负面 Recall 计算正确

## 输出文件

- `src/evaluation/metrics.py` — 指标计算 (F1, AUC, KL, MMD)
- `src/evaluation/evaluator.py` — 完整评估流程
- `src/evaluation/visualize.py` — 可视化工具
- `tests/test_evaluation.py` — 单元测试
