"""模型架构测试"""
import pytest
import torch
from src.models.layers import AdapterLayer, GradientReversalLayer


class TestAdapterLayer:
    """适配器层测试"""

    def test_output_shape(self):
        layer = AdapterLayer(d=768, bottleneck=64)
        x = torch.randn(4, 768)
        out = layer(x)
        assert out.shape == (4, 768)

    def test_residual_connection(self):
        """初始化时接近恒等映射"""
        layer = AdapterLayer(d=768, bottleneck=64)
        x = torch.randn(4, 768)
        out = layer(x)
        # 初始化为零, 残差连接使输出接近输入
        assert torch.allclose(out, x, atol=1e-5)


class TestGradientReversalLayer:
    """梯度反转层测试"""

    def test_forward_identity(self):
        grl = GradientReversalLayer(alpha=1.0)
        x = torch.randn(4, 768, requires_grad=True)
        out = grl(x)
        assert torch.allclose(out, x)

    def test_backward_negates(self):
        grl = GradientReversalLayer(alpha=1.0)
        x = torch.randn(4, 768, requires_grad=True)
        out = grl(x)
        out.sum().backward()
        assert torch.allclose(x.grad, -torch.ones_like(x))


class TestSentimentDomainAdaptModel:
    """主模型测试 (需要下载权重)"""

    @pytest.mark.skip(reason="需要下载 BERT 权重, 慢")
    def test_forward_shape(self, cfg):
        from src.models import SentimentDomainAdaptModel
        model = SentimentDomainAdaptModel(cfg)
        input_ids = torch.randint(0, 1000, (4, 128))
        attention_mask = torch.ones(4, 128, dtype=torch.long)
        feat, logit = model(input_ids, attention_mask)
        assert feat.shape == (4, 768)
        assert logit.shape == (4,)

    @pytest.mark.skip(reason="需要下载 BERT 权重")
    def test_predict_proba_range(self, cfg):
        from src.models import SentimentDomainAdaptModel
        model = SentimentDomainAdaptModel(cfg)
        input_ids = torch.randint(0, 1000, (4, 128))
        attention_mask = torch.ones(4, 128, dtype=torch.long)
        probs = model.predict_proba(input_ids, attention_mask)
        assert (probs >= 0).all() and (probs <= 1).all()
