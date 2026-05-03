"""闭式高斯 KL 散度"""
import torch


def gaussian_kl(
    mu_s: torch.Tensor,
    var_s: torch.Tensor,
    mu_t: torch.Tensor,
    var_t: torch.Tensor,
    eps: float = 1e-8,
) -> torch.Tensor:
    """闭式 KL(N_s ‖ N_t), 对角协方差

    KL = 0.5 * [ tr(Σ_t⁻¹ Σ_s) + (μ_t-μ_s)ᵀ Σ_t⁻¹ (μ_t-μ_s) - d + log(|Σ_t|/|Σ_s|) ]

    Args:
        mu_s: 源域均值 (d,)
        var_s: 源域方差 (d,)
        mu_t: 目标域均值 (d,)
        var_t: 目标域方差 (d,)
        eps: 数值稳定项

    Returns:
        scalar KL divergence
    """
    d = mu_s.shape[0]

    var_t_safe = var_t + eps
    var_s_safe = var_s + eps

    # tr(Σ_t⁻¹ Σ_s)
    tr_term = (var_s_safe / var_t_safe).sum()

    # (μ_t-μ_s)ᵀ Σ_t⁻¹ (μ_t-μ_s)
    mahal_term = ((mu_t - mu_s) ** 2 / var_t_safe).sum()

    # log(|Σ_t|/|Σ_s|)
    log_det_term = (var_t_safe.log() - var_s_safe.log()).sum()

    return 0.5 * (tr_term + mahal_term - d + log_det_term)
