"""文本清洗模块 — 处理社交媒体文本噪声"""
import re
from typing import Optional


class TextCleaner:
    """社交媒体文本清洗器

    处理: @mention, #话题#, URL, 表情, 重复字符, 广告文本
    """

    PATTERNS = {
        "mention": re.compile(r"@\w+"),
        "hashtag": re.compile(r"#([^#]+)#"),
        "url": re.compile(r"https?://\S+"),
        "emoji": re.compile("[\U00010000-\U0010ffff]", flags=re.UNICODE),
        "repeat": re.compile(r"(.)\1{4,}"),
        "ad": re.compile(r"(转发微博|转发抽奖|关注并转发|点击链接)"),
        "whitespace": re.compile(r"\s+"),
    }

    def __init__(self, remove_emoji: bool = True, remove_hashtag: bool = False):
        self.remove_emoji = remove_emoji
        self.remove_hashtag = remove_hashtag

    def clean(self, text: str) -> str:
        if not text or not text.strip():
            return ""

        text = self.PATTERNS["mention"].sub("", text)
        text = self.PATTERNS["url"].sub("", text)
        text = self.PATTERNS["ad"].sub("", text)

        if self.remove_emoji:
            text = self.PATTERNS["emoji"].sub("", text)

        if self.remove_hashtag:
            text = self.PATTERNS["hashtag"].sub("", text)
        else:
            text = self.PATTERNS["hashtag"].sub(r"\1", text)

        # 重复字符压缩
        text = self.PATTERNS["repeat"].sub(r"\1\1", text)
        text = self.PATTERNS["whitespace"].sub(" ", text).strip()

        return text

    def is_noise(self, text: str) -> bool:
        """判断文本是否为纯噪声 (无有效内容)"""
        cleaned = self.clean(text)
        return len(cleaned) < 2
