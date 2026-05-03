"""Hydra 入口 — 跨域情感分析实验"""
import logging
from pathlib import Path

import hydra
from omegaconf import DictConfig, OmegaConf

from src.utils.seed import set_seed

logger = logging.getLogger(__name__)


@hydra.main(config_path="configs", config_name="config", version_base=None)
def main(cfg: DictConfig):
    # 1. 设置随机种子
    set_seed(cfg.seed)
    logger.info(f"Config:\n{OmegaConf.to_yaml(cfg)}")

    # 2. 初始化 W&B
    try:
        import wandb
        wandb.init(
            project=cfg.project,
            config=OmegaConf.to_container(cfg, resolve=True),
            reinit=True,
        )
    except Exception as e:
        logger.warning(f"W&B init failed: {e}. Continuing without W&B.")

    # 3. 数据
    from src.data import CrossDomainDataModule
    datamodule = CrossDomainDataModule(cfg.data)
    datamodule.setup()

    # 4. 模型
    if cfg.model.name == "tfidf_lr":
        return run_tfidf_baseline(datamodule, cfg)

    from src.models import SentimentDomainAdaptModel
    model = SentimentDomainAdaptModel(cfg.model)

    # 5. 训练
    from src.training import Trainer
    trainer = Trainer(model, datamodule, cfg.train)
    checkpoint = trainer.fit()

    # 6. 评估
    from src.evaluation import Evaluator
    evaluator = Evaluator(model)
    # metrics = evaluator.evaluate(loader_s_test, loader_t_val, loader_t_test)

    # 7. 保存
    output_dir = Path(hydra.utils.get_original_cwd()) / "checkpoints"
    output_dir.mkdir(exist_ok=True)
    # torch.save(checkpoint, output_dir / f"checkpoint_seed{cfg.seed}.pt")

    logger.info("Experiment complete!")

    try:
        import wandb
        wandb.finish()
    except Exception:
        pass


def run_tfidf_baseline(datamodule, cfg):
    """E0: TF-IDF + Logistic Regression 基线"""
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import f1_score

    logger.info("Running TF-IDF + LR baseline (E0)...")

    # TODO: 实现 TF-IDF + LR 基线
    # 1. 在源域上 fit TF-IDF
    # 2. 训练 LR
    # 3. 在目标域上评估

    logger.info("TF-IDF baseline complete.")


if __name__ == "__main__":
    main()
