# Agent-3: Training Loop — KL 领域对齐训练 ⭐ 核心

## Task

实现 `src/training/trainer.py` 和 `src/training/kl_loss.py`。

## Loss 分解

```
L_total = L_sup + λ · L_kl + λ_adv · L_adv

L_sup = BCEWithLogitsLoss(logits_t, y_t)
L_kl  = KL( N(μ_s, Σ_s) ‖ N(μ_t, Σ_t) )   # 闭式高斯 KL
L_adv = BCE(domain_pred, domain_label)         # 可选: DANN 对抗损失
```

### 闭式高斯 KL (dim d)

```
KL = 0.5 * [ tr(Σ_t⁻¹ Σ_s) + (μ_t-μ_s)ᵀ Σ_t⁻¹ (μ_t-μ_s) - d + log(|Σ_t|/|Σ_s|) ]
```

对角协方差假设下向量化实现:

```python
def gaussian_kl(mu_s, var_s, mu_t, var_t, eps=1e-8):
    """
    闭式 KL(N_s ‖ N_t), 对角协方差
    Args:
        mu_s, mu_t: (d,)
        var_s, var_t: (d,)  # 方差, 非标准差
    Returns:
        scalar KL divergence
    """
    d = mu_s.shape[0]
    var_t_safe = var_t + eps
    var_s_safe = var_s + eps

    tr_term = (var_s_safe / var_t_safe).sum()
    mahal_term = ((mu_t - mu_s) ** 2 / var_t_safe).sum()
    log_det_term = (var_t_safe.log() - var_s_safe.log()).sum()

    return 0.5 * (tr_term + mahal_term - d + log_det_term)
```

## EMA 特征统计 (关键)

N_t 太小导致 batch 统计噪声大，EMA 平滑 μ_t / σ_t 是稳定训练的关键:

```python
class EMAMeter:
    """指数移动平均特征统计"""
    def __init__(self, decay=0.99):
        self.decay = decay
        self.mu = None
        self.var = None

    def update(self, feat: Tensor):
        """feat: (batch, d)"""
        batch_mu = feat.mean(0)
        batch_var = feat.var(0) + 1e-8
        if self.mu is None:
            self.mu = batch_mu
            self.var = batch_var
        else:
            self.mu = self.decay * self.mu + (1 - self.decay) * batch_mu
            self.var = self.decay * self.var + (1 - self.decay) * batch_var
```

## Training Step (伪代码)

```python
for batch_s, batch_t in zip(loader_s, cycle(loader_t)):
    # forward
    feat_s, _       = model(batch_s.input_ids, batch_s.attention_mask)
    feat_t, logit_t = model(batch_t.input_ids, batch_t.attention_mask)

    # 特征统计 (EMA)
    ema_s.update(feat_s.detach())
    ema_t.update(feat_t.detach())

    # 监督损失
    L_sup = bce(logit_t, batch_t.labels.float())
    if cfg.use_pos_weight:
        L_sup = bce_with_weight(logit_t, batch_t.labels.float(), datamodule.pos_weight)

    # KL 对齐损失
    L_kl = gaussian_kl(ema_s.mu, ema_s.var, ema_t.mu, ema_t.var)

    # 对抗损失 (可选, E5)
    if cfg.use_adversarial:
        domain_pred = discriminator(torch.cat([feat_s, feat_t]))
        L_adv = bce(domain_pred, domain_labels)
    else:
        L_adv = 0.0

    # 总损失
    loss = L_sup + cfg.lambda_kl * L_kl + cfg.lambda_adv * L_adv

    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
    optimizer.step()
    optimizer.zero_grad()
```

> **Note**: `cycle(loader_t)` 因为 N_t ≪ N_s — 目标域 loader 每个 epoch 重置多次。

## Hyperparameters (Hydra `configs/train/baseline.yaml`)

```yaml
optimizer: adamw
lr: 2e-5              # backbone
lr_head: 1e-3         # head (param group)
lr_adapter: 5e-4      # adapter (param group)
weight_decay: 0.01
epochs: 30
lambda_kl: 0.1        # SWEEP: [0.0, 0.01, 0.1, 0.5, 1.0]
lambda_adv: 0.1       # 仅 E5 使用
warmup_epochs: 2
grad_clip: 1.0
ema_decay: 0.99
freeze_layers: 6      # 前 6 层冻结 (BERT 前 50%)
```

## 冻结策略 (来自开题报告)

```python
# Phase 1 (W1-2, warmup): 冻结前 50% 层, 仅训练 head + adapter
# Phase 2 (W3+): 解冻全部, 双学习率
for name, param in model.encoder.named_parameters():
    layer_id = int(name.split('.')[2]) if 'layer' in name else 0
    if layer_id < cfg.freeze_layers:
        param.requires_grad = not is_warmup
```

## Logging (W&B)

Per-step: `loss`, `L_sup`, `L_kl`, `L_adv`, `lr`
Per-epoch: `val/F1_s`, `val/F1_t`, `val/ΔF1`, `val/τ*`, `val/AUC`

## Early Stopping

Monitor `val/ΔF1` (lower is better) with patience=5。

## Tests (`tests/test_training.py`)

- [ ] KL == 0 when (μ_s, Σ_s) == (μ_t, Σ_t)
- [ ] KL > 0 when distributions differ
- [ ] EMA meter 正确平滑
- [ ] Loss 在确定性 toy 数据集上下降
- [ ] 梯度裁剪生效
- [ ] 冻结策略正确切换

## 输出文件

- `src/training/trainer.py` — 主训练循环
- `src/training/kl_loss.py` — 闭式高斯 KL
- `src/training/ema.py` — EMA 特征统计
- `src/training/losses.py` — 组合损失
- `tests/test_training.py` — 单元测试
