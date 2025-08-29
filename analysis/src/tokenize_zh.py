"""
Advanced Chinese Tokenization and Text Processing Module
Comprehensive support for Chinese linguistic structures: single characters, two-character words, 
three-character phrases, idioms, multi-character phrases, and Chinese-English mixed content.
"""

import os
import re
import pickle
import hashlib
import math
from pathlib import Path
from typing import List, Set, Optional, Tuple, Dict, Counter
from collections import defaultdict, Counter
import warnings

import jieba
import jieba.analyse
from tqdm import tqdm


class ChineseTokenizer:
    """Advanced Chinese Tokenizer - Comprehensive Linguistic Structure Support"""
    
    def __init__(self, userdict_path: Optional[str] = None, 
                 stopwords_path: Optional[str] = None,
                 extra_stopwords: Optional[List[str]] = None,
                 cache_dir: str = "analysis/.cache",
                 phrase_dict_path: str = "analysis/assets/chinese_phrases.txt"):
        """
        Initialize Advanced Chinese Tokenizer
        
        Args:
            userdict_path: Path to user dictionary
            stopwords_path: Path to stopwords file  
            extra_stopwords: Additional stopwords list
            cache_dir: Cache directory
            phrase_dict_path: Chinese phrases dictionary path
        """
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        # Load Chinese phrase dictionary (enhance vocabulary recognition)
        self.phrase_dict = self._load_phrase_dict(phrase_dict_path)
        print(f"✅ Loaded Chinese phrase dictionary: {len(self.phrase_dict)} phrases")
        
        # Load user dictionary
        if userdict_path and os.path.exists(userdict_path):
            jieba.load_userdict(userdict_path)
            print(f"✅ Loaded user dictionary: {userdict_path}")
        
        # Load phrase dictionary into jieba
        self._load_phrases_to_jieba()
        
        # Load stopwords
        self.stopwords = self._load_stopwords(stopwords_path, extra_stopwords)
        print(f"✅ Loaded stopwords: {len(self.stopwords)} words")
        
        # Configure jieba for precise mode
        try:
        except Exception:
            # Fallback to default mode if paddle is not available
            warnings.warn("Paddle mode unavailable, using default tokenization mode")
            pass
        
        # Statistical tracking
        self.stats = {
            'total_tokens': 0,
            'single_char_tokens': 0,
            'two_char_tokens': 0,
            'three_char_tokens': 0,
            'four_char_tokens': 0,
            'multi_char_tokens': 0,
            'english_tokens': 0,
            'phrase_tokens': 0,
            'ngram_tokens': 0
        }
        
    def _load_phrase_dict(self, phrase_dict_path: str) -> Dict[str, int]:
        """Load Chinese phrase dictionary"""
        phrase_dict = {}
        
        if not os.path.exists(phrase_dict_path):
            print(f"⚠️ Phrase dictionary file not found: {phrase_dict_path}")
            return phrase_dict
        
        try:
            with open(phrase_dict_path, 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    if not line or line.startswith('#'):
                        continue
                    
                    parts = line.split()
                    if len(parts) >= 2:
                        phrase = parts[0]
                        try:
                            freq = int(parts[1])
                            phrase_dict[phrase] = freq
                        except ValueError:
                            # If frequency is not a number, set default value
                            phrase_dict[phrase] = 100
                    elif len(parts) == 1:
                        # Only phrase without frequency
                        phrase_dict[parts[0]] = 100
        except Exception as e:
            warnings.warn(f"Failed to load phrase dictionary: {e}")
        
        return phrase_dict
    
    def _load_phrases_to_jieba(self):
        """Load phrase dictionary into jieba tokenizer"""
        for phrase, freq in self.phrase_dict.items():
            jieba.add_word(phrase, freq=freq)
        print(f"✅ Loaded {len(self.phrase_dict)} phrases into jieba tokenizer")

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
    
    def tokenize(self, text: str, chinese_only: bool = False, 
                 ngram_max: int = 4, preserve_english: bool = True) -> List[str]:
        """
        Advanced Text Tokenization - Support for Complex Chinese Linguistic Structures
        
        Args:
            text: Input text
            chinese_only: Whether to keep only Chinese (False to keep Chinese-English mixed)
            ngram_max: Maximum n-gram length (1-4 supported for single chars, words, phrases, idioms)
            preserve_english: Whether to preserve English vocabulary
        
        Returns:
            List[str]: Tokenization results with comprehensive linguistic analysis
        """
        if not text:
            return []
        
        # Check cache
        cache_key = self._get_cache_key(text, chinese_only, ngram_max, preserve_english)
        cached_result = self._load_cache(cache_key)
        if cached_result is not None:
            return cached_result
        
        # Preprocess text
        text = self._preprocess_text(text)
        
        # Step 1: Basic jieba tokenization (with loaded phrase dictionary)
        words = jieba.lcut(text, cut_all=False)
        
        # Step 2: Enhanced filtering and classification for different word types
        tokens = []
        for word in words:
            word = word.strip()
            if not word:
                continue
            
            # Classify and process different types of vocabulary
            processed_word = self._process_word(
                word, chinese_only, preserve_english)
            
            if processed_word:
                tokens.extend(processed_word if isinstance(processed_word, list) else [processed_word])
        
        # Step 3: Statistical-driven phrase recognition
        tokens = self._enhance_with_statistical_phrases(tokens, text)
        
        # Step 4: Generate intelligent N-grams (1-4)
        if ngram_max > 1:
            tokens = self._generate_comprehensive_ngrams(tokens, ngram_max)
        
        # Update statistical tracking
        self._update_comprehensive_stats(tokens)
        
        # Cache results
        self._save_cache(cache_key, tokens)
        
        return tokens
    
    def _process_word(self, word: str, chinese_only: bool, preserve_english: bool) -> Optional[List[str]]:
        """处理单个词汇，支持多种语言模式"""
        
        # 停用词过滤
        if word in self.stopwords:
            return None
        
        # 长度过滤
        if len(word) < 1:
            return None
        
        # 纯中文模式
        if chinese_only:
            if not self._is_chinese_or_mixed(word):
                return None
            
            # 单字词特殊处理
            if len(word) == 1:
                if not self._is_meaningful_single_char(word):
                    return None
            
            return [word]
        
        # 混合模式（中英文都保留）
        else:
            # 英文词汇处理
            if self._is_pure_english(word):
                if preserve_english and len(word) >= 2:  # 保留有意义的英文词
                    return [word.lower()]  # 英文转小写
                return None
            
            # 中文词汇处理
            elif self._is_chinese_or_mixed(word):
                if len(word) == 1 and not self._is_meaningful_single_char(word):
                    return None
                return [word]
            
            # 数字和其他字符
            elif self._is_meaningful_number_or_symbol(word):
                return [word]
            
            return None
    
    def _enhance_with_statistical_phrases(self, tokens: List[str], original_text: str) -> List[str]:
        """使用统计方法识别和增强短语"""
        if len(tokens) < 3:
            return tokens
        
        enhanced_tokens = tokens.copy()
        
        # 寻找高频连续词组
        for i in range(len(tokens) - 2):
            # 检查3-字短语
            three_gram = ''.join(tokens[i:i+3])
            if self._is_meaningful_three_char_phrase(three_gram):
                enhanced_tokens.append(three_gram)
            
            # 检查4-字短语（可能是成语）
            if i < len(tokens) - 3:
                four_gram = ''.join(tokens[i:i+4])
                if self._is_meaningful_four_char_phrase(four_gram):
                    enhanced_tokens.append(four_gram)
        
        return enhanced_tokens
    
    def _is_meaningful_three_char_phrase(self, phrase: str) -> bool:
        """判断3字短语是否有意义"""
        # 检查是否在短语词典中
        if phrase in self.phrase_dict:
            return True
        
        # 检查是否符合常见的3字模式
        if len(phrase) == 3 and all(self._is_chinese(char) for char in phrase):
            # 检查是否有语义连贯性
            return self._has_semantic_coherence(phrase)
        
        return False
    
    def _is_meaningful_four_char_phrase(self, phrase: str) -> bool:
        """判断4字短语是否有意义（特别是成语）"""
        # 检查是否在短语词典中
        if phrase in self.phrase_dict:
            return True
        
        # 检查是否符合成语模式
        if len(phrase) == 4 and all(self._is_chinese(char) for char in phrase):
            return self._is_potential_idiom(phrase)
        
        return False
    
    def _has_semantic_coherence(self, phrase: str) -> bool:
        """检查短语是否有语义连贯性"""
        # 简单的语义连贯性检查
        # 避免重复字符组成的无意义短语
        if len(set(phrase)) == 1:  # 全是相同字符
            return False
        
        # 避免全是功能词的组合
        function_chars = {'的', '了', '在', '是', '有', '和', '就', '不', '都', '一', '也', '很', '到'}
        if all(char in function_chars for char in phrase):
            return False
        
        return True
    
    def _is_potential_idiom(self, phrase: str) -> bool:
        """判断是否可能是成语"""
        # 成语通常有以下特征：
        # 1. 四个汉字
        # 2. 不包含常见的功能词
        # 3. 字符多样性好
        
        if len(phrase) != 4:
            return False
        
        # 字符多样性检查
        if len(set(phrase)) < 3:  # 至少3个不同字符
            return False
        
        # 避免包含太多功能词
        function_chars = {'的', '了', '在', '是', '有', '和', '就', '不', '都', '一', '也', '很', '到', '说', '要', '去', '会', '着', '看', '好'}
        function_count = sum(1 for char in phrase if char in function_chars)
        if function_count > 1:  # 功能词不超过1个
            return False
        
        return True
    
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
    
    def _is_chinese_or_mixed(self, word: str) -> bool:
        """判断是否包含中文或中英混合"""
        return bool(re.search(r'[\u4e00-\u9fff]', word))
    
    def _is_pure_english(self, word: str) -> bool:
        """判断是否为纯英文"""
        return bool(re.match(r'^[a-zA-Z]+$', word))
    
    def _is_meaningful_number_or_symbol(self, word: str) -> bool:
        """判断数字或符号是否有意义"""
        # 保留有意义的数字（年份、版本号等）
        if re.match(r'^\d{4}$', word):  # 年份
            return True
        if re.match(r'^\d+\.\d+$', word):  # 版本号
            return True
        if re.match(r'^v\d+', word, re.I):  # 版本标识
            return True
        return False
    
    def _is_meaningful_single_char(self, char: str) -> bool:
        """判断单字是否有意义，值得保留 - 扩展版"""
        # 扩展的有意义单字集合
        meaningful_chars = {
            # 人物称谓与关系
            '人', '我', '你', '他', '她', '它', '谁', '家', '子', '父', '母', '友', '师', '生', '客', '主', '君', '臣',
            
            # 时间与空间
            '年', '月', '日', '时', '天', '夜', '今', '昨', '明', '前', '后', '上', '下', '左', '右', 
            '东', '西', '南', '北', '中', '内', '外', '里', '边', '间', '处', '方', '向', '面',
            
            # 情感与心理
            '心', '情', '爱', '喜', '怒', '忧', '思', '恐', '惊', '痛', '乐', '悲', '哭', '笑', '愁', '怨',
            '恨', '怜', '慈', '悯', '敬', '崇', '仰', '羡', '妒', '惧', '疑', '信', '望', '念',
            
            # 道德与价值
            '道', '理', '义', '礼', '智', '信', '仁', '德', '善', '恶', '真', '假', '美', '丑', '正', '邪',
            '忠', '孝', '廉', '耻', '勇', '勤', '俭', '让', '谦', '诚', '纯', '朴',
            
            # 自然元素
            '天', '地', '山', '水', '火', '木', '金', '土', '风', '雨', '雪', '云', '海', '河', '湖', '江',
            '花', '草', '树', '叶', '果', '根', '枝', '石', '沙', '泥', '光', '影', '月', '星', '日',
            
            # 动作与状态
            '走', '跑', '飞', '坐', '立', '卧', '看', '听', '说', '读', '写', '学', '教', '做', '来', '去',
            '给', '拿', '取', '放', '开', '关', '进', '出', '入', '返', '回', '归', '留', '停', '行', '止',
            
            # 数量与程度
            '一', '二', '三', '四', '五', '六', '七', '八', '九', '十', '百', '千', '万', '亿', '零',
            '多', '少', '全', '半', '无', '有', '空', '满', '深', '浅', '厚', '薄',
            
            # 形容词性概念
            '大', '小', '高', '低', '长', '短', '宽', '窄', '粗', '细', '胖', '瘦', '老', '少', '新', '旧',
            '好', '坏', '美', '丑', '净', '脏', '亮', '暗', '明', '昏', '清', '浊', '冷', '热', '温', '凉',
            '快', '慢', '急', '缓', '强', '弱', '硬', '软', '尖', '钝', '圆', '方', '直', '弯',
            
            # 抽象概念
            '法', '规', '制', '度', '则', '律', '条', '款', '章', '节', '段', '句', '词', '字', '音', '声',
            '色', '形', '象', '意', '思', '想', '念', '识', '知', '觉', '感', '受', '验', '历', '程', '路',
            
            # 社会文化
            '国', '民', '族', '群', '社', '会', '团', '党', '政', '府', '官', '吏', '兵', '将', '王', '帝',
            '文', '化', '艺', '术', '诗', '书', '画', '乐', '歌', '舞', '戏', '剧', '技', '能', '巧', '工',
            
            # 现代概念
            '网', '电', '机', '车', '船', '飞', '科', '技', '数', '码', '信', '息', '据', '库', '系', '统',
            '软', '硬', '件', '务', '业', '商', '贸', '易', '市', '场', '价', '值', '钱', '财', '富',
            
            # 学术研究
            '研', '究', '学', '术', '论', '文', '著', '作', '史', '传', '记', '录', '档', '案', '料', '源'
        }
        return char in meaningful_chars
    
    def _generate_comprehensive_ngrams(self, tokens: List[str], max_n: int) -> List[str]:
        """Comprehensive N-gram Generation (1-4) - Based on Linguistic Rules and Statistical Features"""
        result = tokens.copy()
        
        for n in range(2, max_n + 1):
            ngrams_added = 0
            for i in range(len(tokens) - n + 1):
                ngram_tokens = tokens[i:i+n]
                
                # Intelligent validation of N-gram value
                if self._is_valuable_ngram(ngram_tokens, n):
                    # Intelligent connection strategy
                    ngram = self._create_intelligent_ngram(ngram_tokens)
                    if ngram and ngram not in result:  # Avoid duplicates
                        result.append(ngram)
                        ngrams_added += 1
        
        return result
    
    def _is_valuable_ngram(self, tokens: List[str], n: int) -> bool:
        """判断N-gram是否有价值"""
        if not tokens or len(tokens) != n:
            return False
        
        # 避免重复词
        if len(set(tokens)) != len(tokens):
            return False
        
        # 避免全停用词组合
        if all(token in self.stopwords for token in tokens):
            return False
        
        # 避免纯功能词组合
        function_words = {
            '的', '了', '在', '是', '有', '和', '就', '不', '都', '一', '也', '很', '到', '说', '要', '去',
            '会', '着', '没有', '看', '好', '自己', '这', '那', '这个', '那个', '什么', '怎么', '为什么'
        }
        function_count = sum(1 for token in tokens if token in function_words)
        if function_count > n // 2:  # 功能词不能超过一半
            return False
        
        # 检查语义连贯性
        return self._check_semantic_coherence(tokens)
    
    def _check_semantic_coherence(self, tokens: List[str]) -> bool:
        """检查词汇组合的语义连贯性"""
        # 简单的语义连贯性检查
        
        # 1. 检查是否包含实义词
        content_words = 0
        for token in tokens:
            if len(token) > 1 or self._is_meaningful_single_char(token):
                content_words += 1
        
        if content_words == 0:
            return False
        
        # 2. 检查中英文混合的合理性
        chinese_count = sum(1 for token in tokens if self._is_chinese(token))
        english_count = sum(1 for token in tokens if self._is_pure_english(token))
        
        # 如果有中英混合，应该是有意义的组合
        if chinese_count > 0 and english_count > 0:
            # 例如：machine_learning、AI_技术 等
            return True
        
        return True
    
    def _create_intelligent_ngram(self, tokens: List[str]) -> Optional[str]:
        """智能创建N-gram字符串"""
        if not tokens:
            return None
        
        # 策略1：纯中文单字组合直接连接
        if all(len(token) == 1 and self._is_chinese(token) for token in tokens):
            return ''.join(tokens)
        
        # 策略2：包含英文或多字词用下划线连接
        elif any(self._is_pure_english(token) or len(token) > 1 for token in tokens):
            return '_'.join(tokens)
        
        # 策略3：混合情况的智能处理
        else:
            # 根据词汇类型决定连接方式
            result_parts = []
            current_chinese_chars = []
            
            for token in tokens:
                if len(token) == 1 and self._is_chinese(token):
                    current_chinese_chars.append(token)
                else:
                    # 遇到多字词或英文，先处理累积的中文字符
                    if current_chinese_chars:
                        result_parts.append(''.join(current_chinese_chars))
                        current_chinese_chars = []
                    result_parts.append(token)
            
            # 处理末尾的中文字符
            if current_chinese_chars:
                result_parts.append(''.join(current_chinese_chars))
            
            return '_'.join(result_parts) if len(result_parts) > 1 else result_parts[0]
    
    def _update_comprehensive_stats(self, tokens: List[str]):
        """Update comprehensive tokenization statistics"""
        self.stats['total_tokens'] += len(tokens)
        
        for token in tokens:
            # Classify by character length and type
            if self._is_pure_english(token):
                self.stats['english_tokens'] += 1
            elif self._is_chinese(token):
                char_count = len([c for c in token if self._is_chinese(c)])
                if char_count == 1:
                    self.stats['single_char_tokens'] += 1
                elif char_count == 2:
                    self.stats['two_char_tokens'] += 1
                elif char_count == 3:
                    self.stats['three_char_tokens'] += 1
                elif char_count == 4:
                    self.stats['four_char_tokens'] += 1
                else:
                    self.stats['multi_char_tokens'] += 1
            
            # Check if it's a phrase from dictionary
            if token in self.phrase_dict:
                self.stats['phrase_tokens'] += 1
            
            # Check if it's an n-gram (contains underscore)
            if '_' in token:
                self.stats['ngram_tokens'] += 1
    
    def get_stats(self) -> Dict[str, int]:
        """获取分词统计信息"""
        return self.stats.copy()
    
    def _get_cache_key(self, text: str, chinese_only: bool, ngram_max: int, preserve_english: bool = True) -> str:
        """生成缓存键"""
        content = f"{text}_{chinese_only}_{ngram_max}_{preserve_english}"
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