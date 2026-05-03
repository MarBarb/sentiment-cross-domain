# Agent-2: Model Architecture — 跨域情感分类模型

## Task

实现 `src/models/model.py`，包含特征提取器 φ + 分类头。

## 设计

```
x → Tokenizer → φ(x) ∈ R^d → head → logit ∈ R
```

### φ (encoder): 可配置 backbone

| Backbone | 来源 | 参数量 | 适用场景 |
|----------|------|--------|---------|
| `bert-base-uncased` | HuggingFace | 110M | 默认 (E1-E5) |
| `roberta-base` | HuggingFace | 125M | backbone 替换对比 (E6) |
| TF-IDF + LR (sklearn) | — | — | 浅层基线 (E0, 不在此模块) |

### Head

```python
nn.Linear(d, 1)  # binary, sigmoid at inference
```

### PEFT 适配器 (M1)

```python
# 方案 A: LoRA
from peft import LoraConfig, get_peft_model
lora_config = LoraConfig(
    r=8, lora_alpha=16, target_modules=["query", "value"],
    lora_dropout=0.1, bias="none"
)

# 方案 B: Adapter (小型 bottleneck)
class AdapterLayer(nn.Module):
    def __init__(self, d, bottleneck=64):
        self.down = nn.Linear(d, bottleneck)
        self.up = nn.Linear(bottleneck, d)
        self.act = nn.GELU()
```

## Required API

```python
class SentimentDomainAdaptModel(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(cfg.backbone)
        self.dropout = nn.Dropout(cfg.dropout)
        self.head = nn.Linear(self.encoder.config.hidden_size, 1)
        # 可选: 冻结策略
        if cfg.freeze_layers > 0:
            self._freeze_first_n(cfg.freeze_layers)

    def forward(self, input_ids, attention_mask) -> tuple[Tensor, Tensor]:
        """Returns (features φ(x), logits)"""
        out = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        feat = out.last_hidden_state[:, 0, :]  # [CLS] token
        feat = self.dropout(feat)
        logit = self.head(feat).squeeze(-1)
        return feat, logit

    @torch.no_grad()
    def predict_proba(self, input_ids, attention_mask) -> Tensor:
        _, logit = self.forward(input_ids, attention_mask)
        return torch.sigmoid(logit)

    def _freeze_first_n(self, n):
        """冻结 encoder 前 n 层"""
```

## Why expose features?

KL 项需要 P_s(φ(x)) 和 P_t(φ(x))。我们将其建模为 **多元高斯分布**:
- μ, σ² 通过 batch 统计估计 (EMA 平滑)
- 闭式 KL 在 Agent-3 中实现

## Init Strategy

- 加载预训练 backbone 权重
- **冻结策略** (来自开题报告):
  - W1-2: 冻结前 50% encoder 层，仅训练 head + adapter (warm-up)
  - W3+: 解冻全部层，backbone lr=2e-5，head lr=1e-3
- LoRA: 仅训练 <1% 参数，适合小目标域

## 对抗域判别器 (M2, 可选)

```python
class DomainDiscriminator(nn.Module):
    """DANN 风格域判别器 + 梯度反转层"""
    def __init__(self, feat_dim, hidden=256):
        self.net = nn.Sequential(
            nn.Linear(feat_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 1)
        )
    def forward(self, feat):
        return self.net(GradientReversalLayer.apply(feat))
```

## Tests (`tests/test_model.py`)

- [ ] forward 输出 shape 正确: (batch, d) 和 (batch,)
- [ ] predict_proba 输出在 [0, 1]
- [ ] 冻结策略正确: 前 N 层 requires_grad=False
- [ ] LoRA 参数量 < 总参数量 1%
- [ ] 域判别器输出 shape 正确

## 输出文件

- `src/models/model.py` — 主模型
- `src/models/domain_discriminator.py` — 域判别器
- `src/models/layers.py` — Adapter 层、梯度反转层
- `tests/test_model.py` — 单元测试
