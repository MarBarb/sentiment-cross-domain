"""可视化工具"""
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE


def plot_tsne(feats_s, feats_t, save_path="tsne_features.png"):
    """t-SNE 可视化源域 vs 目标域特征分布"""
    feats = np.concatenate([feats_s, feats_t], axis=0)
    labels = ["Source"] * len(feats_s) + ["Target"] * len(feats_t)

    tsne = TSNE(n_components=2, random_state=42)
    feats_2d = tsne.fit_transform(feats)

    plt.figure(figsize=(10, 8))
    for label, color in [("Source", "blue"), ("Target", "red")]:
        mask = [l == label for l in labels]
        plt.scatter(feats_2d[mask, 0], feats_2d[mask, 1], c=color, label=label, alpha=0.5)

    plt.legend()
    plt.title("t-SNE: Source vs Target Features")
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()


def plot_threshold_scan(probs, labels, save_path="threshold_scan.png"):
    """τ vs F1 曲线"""
    from sklearn.metrics import f1_score

    taus = np.arange(0.05, 0.95, 0.01)
    f1s = [f1_score(labels, (probs >= t).astype(int), average="macro") for t in taus]

    plt.figure(figsize=(8, 6))
    plt.plot(taus, f1s)
    best_tau = taus[np.argmax(f1s)]
    plt.axvline(best_tau, color="red", linestyle="--", label=f"τ*={best_tau:.2f}")
    plt.xlabel("Threshold τ")
    plt.ylabel("Macro-F1")
    plt.title("Threshold Scan")
    plt.legend()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()


def plot_loss_curves(log_path, save_path="loss_curves.png"):
    """训练损失曲线"""
    # TODO: 从 W&B 或 JSON log 解析并绘图
    pass
