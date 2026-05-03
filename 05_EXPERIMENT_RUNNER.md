# Agent-5: Experiment Orchestration — 实验编排

## Task

实现 `run.py` (Hydra 入口) + 消融实验脚本 + W&B sweep 配置。

## Hydra Layout

```
configs/
├── config.yaml                  # 默认配置
├── data/
│   ├── sst2.yaml               # SST-2 源域
│   ├── imdb.yaml               # IMDB 源域
│   └── social.yaml             # 社会事件评论目标域
├── model/
│   ├── bert_base.yaml          # BERT-base
│   ├── roberta_base.yaml       # RoBERTa-base (E6)
│   └── tfidf_lr.yaml           # TF-IDF+LR 基线 (E0)
├── train/
│   ├── baseline.yaml           # 基础训练配置
│   ├── kl_align.yaml           # KL 对齐训练
│   └── adversarial.yaml        # DANN 对抗训练
└── experiment/
    ├── source_only.yaml         # E1: λ=0, 无目标域监督
    ├── finetune_only.yaml       # E3: λ=0, 有目标域监督 + adapter
    ├── weak_sup_only.yaml       # E2: 弱监督伪标
    ├── kl_align.yaml            # E4: 弱监督 + KL 对齐
    ├── full_method.yaml         # E5: 完整方案
    └── roberta_full.yaml        # E6: RoBERTa + 完整方案
```

## run.py (Hydra 入口)

```python
import hydra
from omegaconf import DictConfig

@hydra.main(config_path="configs", config_name="config", version_base=None)
def main(cfg: DictConfig):
    # 1. 设置随机种子
    set_seed(cfg.seed)

    # 2. 初始化 W&B
    wandb.init(project=cfg.project, config=omegaconf.to_container(cfg))

    # 3. 数据
    datamodule = CrossDomainDataModule(cfg.data)
    datamodule.setup()

    # 4. 模型
    if cfg.model.name == "tfidf_lr":
        return run_tfidf_baseline(datamodule, cfg)
    model = SentimentDomainAdaptModel(cfg.model)

    # 5. 训练
    trainer = Trainer(model, datamodule, cfg.train)
    checkpoint = trainer.fit()

    # 6. 评估
    metrics = evaluate(checkpoint, datamodule)
    wandb.log(metrics)

    # 7. 保存
    save_checkpoint(checkpoint, cfg, metrics)

if __name__ == "__main__":
    main()
```

## 消融实验配置

### E0: TF-IDF + LR (浅层基线)
```yaml
# configs/experiment/source_only.yaml 中 model 部分
model:
  name: tfidf_lr
  max_features: 50000
```

### E1: BERT zero-shot (跨域衰减基准)
```yaml
train:
  lambda_kl: 0.0
  lambda_adv: 0.0
  epochs: 0  # 不训练, 直接评估预训练模型
```

### E5: 完整方案
```yaml
train:
  lambda_kl: 0.1
  lambda_adv: 0.1
  use_weak_supervision: true
  use_adapter: true
  use_adversarial: true
```

## Sweeps (W&B)

```yaml
# configs/sweeps/kl_lambda.yaml
program: run.py
method: bayes
metric: { name: val/deltaF1, goal: minimize }
parameters:
  train.lambda_kl: { values: [0.0, 0.01, 0.1, 0.5, 1.0] }
  train.lr:        { min: 1e-6, max: 1e-3, distribution: log_uniform }
  data.batch_size_t: { values: [8, 16, 32] }
  train.ema_decay: { values: [0.95, 0.99, 0.999] }
```

## Reproducibility

```python
def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
```

保存: `checkpoint.pt`, `config.yaml`, `metrics.json`, `wandb_run_id.txt`

## CLI Examples

```bash
# E0: 浅层基线
python run.py experiment=source_only model=tfidf_lr

# E1: BERT zero-shot
python run.py experiment=source_only

# E5: 完整方案 (KL + 弱监督 + Adapter + DANN)
python run.py experiment=full_method train.lambda_kl=0.1

# E6: RoBERTa backbone
python run.py experiment=roberta_full

# 消融 sweep
wandb sweep configs/sweeps/kl_lambda.yaml
wandb agent <sweep_id>

# 单次实验, 指定种子
python run.py experiment=kl_align seed=42

# 三种子批量 (论文用)
for seed in 42 123 456; do
  python run.py experiment=full_method seed=$seed
done
```

## 输出文件

- `run.py` — Hydra 入口
- `configs/` — 所有 YAML 配置
- `scripts/run_ablation.sh` — 批量消融脚本
- `scripts/run_sweep.sh` — W&B sweep 脚本
