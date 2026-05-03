# Agent-1: Data Pipeline — 跨域情感分析数据模块

## Task

实现 `src/data/datamodule.py`，暴露 `CrossDomainDataModule`。

## 数据源

| 数据集 | 来源 | 规模 | 用途 |
|--------|------|------|------|
| SST-2 | HuggingFace `glue/sst2` | ~67k train | 源域训练 |
| IMDB | HuggingFace `imdb` 或 Kaggle | 50k | 源域训练 / 跨域测试 |
| 社会事件评论 | 自建爬虫 (微博/知乎/豆瓣) | 5k+ 有效 | 目标域 |
| 领域词典 | HowNet + 网络用语词典 | — | 特征增强 |

## Requirements

1. 加载 D_s (大规模标注影评) 和 D_t (少量标注 + 大量无标注社会事件评论)
2. 每个训练步返回 **两个** dataloader:
   - `loader_s`: 源域 batch (用于 KL 特征统计)
   - `loader_t`: 目标域 batch (用于监督损失)
3. D_t 分层划分 train/val/test (60/20/20)，N_t 较小
4. 源域和目标域使用相同的 BERT tokenizer 预处理
5. 支持类别不平衡: 计算 `pos_weight` 供 BCE 损失使用

## 数据挑战 (来自开题报告)

1. **概念漂移**: 影评 vs 社会事件评论，词汇和情感触发词分布差异大，预计 KL > 1.0
2. **类别不平衡**: 目标域负面偏重 (负:正 ≈ 2:1 ~ 4:1)
3. **标签噪声**: 目标域无天然标签，少量金标 + 大量银标
4. **文本噪声**: 表情、话题标签 #xxx#、@mention、广告
5. **反讽与隐式情感**: 目标域反讽比例高于源域

## Class Skeleton

```python
class CrossDomainDataModule:
    def __init__(self, cfg):
        self.tokenizer = AutoTokenizer.from_pretrained(cfg.backbone)
        self.max_length = cfg.max_length      # 128 or 256
        self.batch_size_s = cfg.batch_size_s   # e.g. 32
        self.batch_size_t = cfg.batch_size_t   # e.g. 16 (smaller)
        self.seed = cfg.seed

    def setup(self):
        """加载数据集, 划分目标域"""
        self.ds_source = self._load_source()    # SST-2 + IMDB
        self.ds_target_labeled, self.ds_target_unlabeled, self.ds_target_test = \
            self._load_target()                  # 社会事件评论

    def train_loaders(self) -> tuple[DataLoader, DataLoader]:
        """返回 (loader_s, loader_t)"""
        return loader_s, loader_t

    def val_loader(self) -> DataLoader:
        """目标域验证集"""

    def test_loaders(self) -> tuple[DataLoader, DataLoader]:
        """返回 (loader_s_test, loader_t_test)"""

    @property
    def pos_weight(self) -> Tensor:
        """目标域类别不平衡权重, 供 BCEWithLogitsLoss 使用"""

    def _clean_text(self, text: str) -> str:
        """清洗: 去除 @mention、话题标签、广告、重复字符等"""
```

## 文本清洗规则

```python
CLEAN_RULES = {
    "mention":   r"@\w+",           # @某人
    "hashtag":   r"#([^#]+)#",      # #话题#
    "url":       r"https?://\S+",
    "emoji":     r"[\U00010000-\U0010ffff]",
    "repeat":    r"(.)\1{4,}",      # 重复字符 > 4次
    "ad":        r"(转发微博|转发抽奖|关注并转发)",
}
```

## Edge Cases

- N_t < batch_size_t: 使用 `RandomSampler(replacement=True)` 过采样
- 目标域类别不平衡: 计算 `pos_weight` 并通过 `datamodule.pos_weight` 暴露
- 文本过长: BERT tokenizer 截断到 max_length
- 空文本 / 纯噪声文本: 跳过并记录 warning

## Tests (`tests/test_data.py`)

- [ ] 源域/目标域 batch shape 一致
- [ ] D_t train/val/test 无标签泄漏
- [ ] pos_weight 计算正确
- [ ] tokenizer 输出 shape 符合预期
- [ ] 清洗规则正确处理 @mention、话题标签等
- [ ] 小数据集 (N_t < batch_size) 不崩溃

## 输出文件

- `src/data/datamodule.py` — 主模块
- `src/data/cleaner.py` — 文本清洗
- `src/data/datasets.py` — 自定义 Dataset 类
- `tests/test_data.py` — 单元测试
