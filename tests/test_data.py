"""数据管道测试"""
import pytest
from src.data.cleaner import TextCleaner


class TestTextCleaner:
    """文本清洗测试"""

    def setup_method(self):
        self.cleaner = TextCleaner()

    def test_clean_mention(self):
        text = "今天天气真好 @某人 你觉得呢"
        result = self.cleaner.clean(text)
        assert "@某人" not in result
        assert "今天天气真好" in result

    def test_clean_url(self):
        text = "看看这个 https://example.com/path?query=1"
        result = self.cleaner.clean(text)
        assert "https://" not in result
        assert "看看这个" in result

    def test_clean_hashtag(self):
        text = "这个话题很火 #社会热点# 你怎么看"
        result = self.cleaner.clean(text)
        assert "#社会热点#" not in result
        assert "社会热点" in result

    def test_clean_emoji(self):
        text = "好开心😀😀😀"
        result = self.cleaner.clean(text)
        assert "😀" not in result
        assert "好开心" in result

    def test_clean_repeat(self):
        text = "太好了好好好好好好好好好好"
        result = self.cleaner.clean(text)
        assert "好好好好" not in result

    def test_clean_ad(self):
        text = "转发微博 抽奖送手机"
        result = self.cleaner.clean(text)
        assert "转发微博" not in result

    def test_is_noise(self):
        assert self.cleaner.is_noise("") == True
        assert self.cleaner.is_noise("a") == True
        assert self.cleaner.is_noise("今天天气不错") == False

    def test_clean_empty(self):
        assert self.cleaner.clean("") == ""
        assert self.cleaner.clean("   ") == ""


class TestDataModule:
    """数据模块测试 (需要实现后取消跳过)"""

    @pytest.mark.skip(reason="需要实现 _load_source 和 _load_target")
    def test_batch_shape(self):
        pass

    @pytest.mark.skip(reason="需要实现数据加载")
    def test_no_label_leakage(self):
        pass

    @pytest.mark.skip(reason="需要实现数据加载")
    def test_pos_weight(self):
        pass
