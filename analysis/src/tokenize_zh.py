"""
中文分词与文本清洗模块
"""

import os
import re
import pickle
import hashlib
from pathlib import Path
from typing import List, Set, Optional, Tuple
import warnings

import jieba
import jieba.analyse
from tqdm import tqdm


class ChineseTokenizer:
    """中文分词器"""
    
    def __init__(self, userdict_path: Optional[str] = None, 
                 stopwords_path: Optional[str] = None,
                 extra_stopwords: Optional[List[str]] = None,
                 cache_dir: str = "analysis/.cache"):
        """
        初始化分词器
        
        Args:
            userdict_path: 用户词典路径
            stopwords_path: 停用词文件路径  
            extra_stopwords: 额外停用词列表
            cache_dir: 缓存目录
        """
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        # 加载用户词典
        if userdict_path and os.path.exists(userdict_path):
            jieba.load_userdict(userdict_path)
            print(f"已加载用户词典: {userdict_path}")
        
        # 加载停用词
        self.stopwords = self._load_stopwords(stopwords_path, extra_stopwords)
        print(f"已加载停用词: {len(self.stopwords)} 个")
        
        # 设置 jieba 为精确模式
        try:
            jieba.enable_paddle()  # 使用 paddle 模式提高准确率
        except:
            # 如果 paddle 不可用，使用默认模式
            warnings.warn("Paddle 模式不可用，使用默认分词模式")
            pass
        
    def _load_stopwords(self, stopwords_path: Optional[str], 
                       extra_stopwords: Optional[List[str]]) -> Set[str]:
        """加载停用词"""
        stopwords = set()
        
        # 从文件加载
        if stopwords_path and os.path.exists(stopwords_path):
            try:
                with open(stopwords_path, 'r', encoding='utf-8') as f:
                    for line in f:
                        word = line.strip()
                        if word:
                            stopwords.add(word)
            except Exception as e:
                warnings.warn(f"加载停用词文件失败: {e}")
        
        # 添加额外停用词
        if extra_stopwords:
            stopwords.update(extra_stopwords)
        
        # 添加默认停用词
        default_stopwords = {
            '的', '了', '在', '是', '我', '有', '和', '就', '不', '人', 
            '都', '一', '一个', '上', '也', '很', '到', '说', '要', '去', 
            '你', '会', '着', '没有', '看', '好', '自己', '这', '那', '这个',
            '那个', '什么', '怎么', '为什么', '可以', '能够', '应该', '需要',
            '但是', '如果', '因为', '所以', '然后', '还是', '或者', '已经',
            '正在', '可能', '应该', '必须', '当然', '确实', '其实', '特别',
            '非常', '比较', '更加', '最好', '最大', '最小', '第一', '第二',
            '等等', '之类', '以及', '以上', '以下', '之后', '之前', '同时',
            '另外', '此外', '而且', '并且', '或许', '大概', '可能', '似乎'
        }
        stopwords.update(default_stopwords)
        
        return stopwords
    
    def tokenize(self, text: str, chinese_only: bool = True, 
                 ngram_max: int = 1) -> List[str]:
        """
        对文本进行分词
        
        Args:
            text: 输入文本
            chinese_only: 是否只保留中文
            ngram_max: 最大 n-gram 长度
        
        Returns:
            List[str]: 分词结果
        """
        if not text:
            return []
        
        # 检查缓存
        cache_key = self._get_cache_key(text, chinese_only, ngram_max)
        cached_result = self._load_cache(cache_key)
        if cached_result is not None:
            return cached_result
        
        # 预处理
        text = self._preprocess_text(text)
        
        # jieba 分词
        words = jieba.lcut(text, cut_all=False)
        
        # 过滤和清洗
        tokens = []
        for word in words:
            word = word.strip()
            if not word:
                continue
                
            # 中文过滤
            if chinese_only and not self._is_chinese(word):
                continue
            
            # 停用词过滤
            if word in self.stopwords:
                continue
            
            # 智能长度过滤：对于中文，允许有意义的单字词
            if len(word) < 1:
                continue
            
            # 单字词特殊处理：只保留有意义的中文单字
            if len(word) == 1 and chinese_only:
                if not self._is_meaningful_single_char(word):
                    continue
                
            tokens.append(word)
        
        # 生成 n-gram (改进逻辑)
        if ngram_max > 1:
            tokens = self._generate_ngrams_improved(tokens, ngram_max)
        
        # 缓存结果
        self._save_cache(cache_key, tokens)
        
        return tokens
    
    def tokenize_batch(self, texts: List[str], chinese_only: bool = True,
                      ngram_max: int = 1, show_progress: bool = True) -> List[List[str]]:
        """
        批量分词
        
        Args:
            texts: 文本列表
            chinese_only: 是否只保留中文
            ngram_max: 最大 n-gram 长度
            show_progress: 是否显示进度条
        
        Returns:
            List[List[str]]: 分词结果列表
        """
        results = []
        
        if show_progress:
            texts = tqdm(texts, desc="分词处理")
        
        for text in texts:
            tokens = self.tokenize(text, chinese_only, ngram_max)
            results.append(tokens)
        
        return results
    
    def extract_keywords(self, text: str, topk: int = 20, 
                        method: str = "tfidf") -> List[Tuple[str, float]]:
        """
        提取关键词
        
        Args:
            text: 输入文本
            topk: 返回关键词数量
            method: 提取方法，'tfidf' 或 'textrank'
        
        Returns:
            List[Tuple[str, float]]: (关键词, 权重) 列表
        """
        if method == "tfidf":
            keywords = jieba.analyse.extract_tags(text, topK=topk, withWeight=True)
        elif method == "textrank":
            keywords = jieba.analyse.textrank(text, topK=topk, withWeight=True)
        else:
            raise ValueError(f"不支持的方法: {method}")
        
        return keywords
    
    def _preprocess_text(self, text: str) -> str:
        """文本预处理"""
        # 移除HTML标签
        text = re.sub(r'<[^>]+>', '', text)
        # 移除URL
        text = re.sub(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', '', text)
        # 移除邮箱
        text = re.sub(r'\S*@\S*\s?', '', text)
        # 移除多余空白
        text = re.sub(r'\s+', ' ', text)
        # 移除数字和标点（可选）
        # text = re.sub(r'[0-9\W]+', ' ', text)
        
        return text.strip()
    
    def _is_chinese(self, word: str) -> bool:
        """判断是否为中文"""
        return bool(re.search(r'[\u4e00-\u9fff]', word))
    
    def _is_meaningful_single_char(self, char: str) -> bool:
        """判断单字是否有意义，值得保留"""
        # 保留常见的有意义单字
        meaningful_chars = {
            # 人物称谓
            '人', '我', '你', '他', '她', '它', '谁', '家', '子', '父', '母', '友',
            # 时间空间
            '年', '月', '日', '时', '天', '夜', '今', '昨', '明', '前', '后', '上', '下', '左', '右', '东', '西', '南', '北',
            # 情感心理
            '心', '情', '爱', '喜', '怒', '忧', '思', '恐', '惊', '痛', '乐', '悲', '哭', '笑',
            # 抽象概念
            '道', '理', '义', '礼', '智', '信', '仁', '德', '善', '恶', '真', '假', '美', '丑',
            # 自然元素
            '天', '地', '山', '水', '火', '木', '金', '土', '风', '雨', '雪', '云', '海', '河', '花', '草', '树',
            # 动作状态
            '走', '跑', '飞', '坐', '立', '卧', '看', '听', '说', '读', '写', '学', '教', '做', '来', '去', '给', '拿',
            # 数量单位
            '一', '二', '三', '四', '五', '六', '七', '八', '九', '十', '百', '千', '万', '亿',
            # 其他常用字
            '大', '小', '多', '少', '好', '坏', '新', '旧', '长', '短', '高', '低', '快', '慢', '远', '近'
        }
        return char in meaningful_chars
    
    def _generate_ngrams(self, tokens: List[str], max_n: int) -> List[str]:
        """生成 n-gram（旧版本，保留兼容性）"""
        result = tokens.copy()
        
        for n in range(2, max_n + 1):
            for i in range(len(tokens) - n + 1):
                ngram = ''.join(tokens[i:i+n])
                result.append(ngram)
        
        return result
    
    def _generate_ngrams_improved(self, tokens: List[str], max_n: int) -> List[str]:
        """改进的 n-gram 生成，适用于中文分析"""
        result = tokens.copy()
        
        for n in range(2, max_n + 1):
            for i in range(len(tokens) - n + 1):
                # 构建n-gram时考虑语义连贯性
                ngram_tokens = tokens[i:i+n]
                
                # 对于中文，检查是否是有意义的组合
                if self._is_meaningful_ngram(ngram_tokens):
                    # 使用适当的连接方式
                    if all(len(token) == 1 for token in ngram_tokens):
                        # 全部是单字，直接连接
                        ngram = ''.join(ngram_tokens)
                    else:
                        # 包含多字词，用下划线连接保持可读性
                        ngram = '_'.join(ngram_tokens)
                    
                    result.append(ngram)
        
        return result
    
    def _is_meaningful_ngram(self, tokens: List[str]) -> bool:
        """判断n-gram组合是否有意义"""
        if not tokens or len(tokens) < 2:
            return False
        
        # 基本过滤：避免重复词
        if len(set(tokens)) != len(tokens):
            return False
        
        # 避免全部都是功能词的组合
        function_words = {'的', '了', '在', '是', '有', '和', '就', '不', '都', '一', '也', '很', '到'}
        if all(token in function_words for token in tokens):
            return False
        
        # 保留至少包含一个实词的组合
        return True
    
    def _get_cache_key(self, text: str, chinese_only: bool, ngram_max: int) -> str:
        """生成缓存键"""
        content = f"{text}_{chinese_only}_{ngram_max}"
        return hashlib.md5(content.encode('utf-8')).hexdigest()
    
    def _load_cache(self, cache_key: str) -> Optional[List[str]]:
        """加载缓存"""
        cache_file = self.cache_dir / f"{cache_key}.pkl"
        if cache_file.exists():
            try:
                with open(cache_file, 'rb') as f:
                    return pickle.load(f)
            except Exception:
                pass
        return None
    
    def _save_cache(self, cache_key: str, tokens: List[str]):
        """保存缓存"""
        cache_file = self.cache_dir / f"{cache_key}.pkl"
        try:
            with open(cache_file, 'wb') as f:
                pickle.dump(tokens, f)
        except Exception:
            pass


def create_default_stopwords(output_path: str = "analysis/assets/stopwords_zh.txt"):
    """创建默认中文停用词文件"""
    stopwords = [
        # 常用虚词
        '的', '了', '在', '是', '我', '有', '和', '就', '不', '人', '都', '一', '一个',
        '上', '也', '很', '到', '说', '要', '去', '你', '会', '着', '没有', '看', '好',
        '自己', '这', '那', '这个', '那个', '什么', '怎么', '为什么', '可以', '能够',
        
        # 连词介词  
        '但是', '如果', '因为', '所以', '然后', '还是', '或者', '已经', '正在', '可能',
        '应该', '必须', '当然', '确实', '其实', '特别', '非常', '比较', '更加',
        
        # 数词序词
        '第一', '第二', '第三', '最好', '最大', '最小', '等等', '之类', '以及', '以上',
        '以下', '之后', '之前', '同时', '另外', '此外', '而且', '并且', '或许', '大概',
        '似乎', '可能', '肯定', '一定', '也许', '大约', '左右', '上下', '前后',
        
        # 语气词
        '啊', '呀', '哦', '嗯', '哈', '呢', '吧', '吗', '么', '哟', '嘿', '哎',
        
        # 标点符号
        '，', '。', '！', '？', '；', '：', '"', '"', ''', ''', '（', '）', '【', '】',
        '《', '》', '〈', '〉', '——', '…', '、', '·'
    ]
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    with open(output_path, 'w', encoding='utf-8') as f:
        for word in stopwords:
            f.write(word + '\n')
    
    print(f"默认停用词文件已创建: {output_path}")
    return stopwords


if __name__ == "__main__":
    # 创建默认停用词文件
    create_default_stopwords()