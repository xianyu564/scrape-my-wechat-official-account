#!/usr/bin/env python3
"""
Pluggable tokenizers and normalizers with pkuseg primary, jieba fallback
"""

import os
import re
from typing import List, Set, Optional, Protocol
import warnings
import unicodedata

# Try importing pkuseg first, fallback to jieba
PKUSEG_AVAILABLE = False
try:
    import pkuseg
    PKUSEG_AVAILABLE = True
    print("✅ pkuseg available - using as primary tokenizer")
except ImportError:
    print("⚠️  pkuseg not available - falling back to jieba")

import jieba
import jieba.analyse


class TokenizerProtocol(Protocol):
    """Protocol for pluggable tokenizers"""
    
    def tokenize(self, text: str) -> List[str]:
        """Tokenize text into list of tokens"""
        ...


class ChineseNormalizer:
    """Text normalizer for Chinese and mixed-language content"""
    
    @staticmethod
    def normalize(text: str) -> str:
        """
        Normalize text with Chinese-first approach
        - Convert full-width to half-width
        - Convert English to lowercase
        - Preserve hyphens, underscores, numbers
        """
        if not text:
            return ""
            
        # Convert full-width to half-width
        normalized = unicodedata.normalize('NFKC', text)
        
        # Replace full-width punctuation with half-width
        replacements = {
            '，': ',', '。': '.', '？': '?', '！': '!', 
            '：': ':', '；': ';', '"': '"', '"': '"',
            ''': "'", ''': "'", '（': '(', '）': ')',
            '【': '[', '】': ']', '《': '<', '》': '>',
            '、': ',', '　': ' '  # Full-width space to half-width
        }
        
        for full, half in replacements.items():
            normalized = normalized.replace(full, half)
        
        # Normalize English: preserve machine-learning, AIGC_2025 patterns
        # Convert to lowercase but preserve technical terms structure
        def normalize_english(match):
            word = match.group(0)
            # Preserve structure but convert to lowercase
            return word.lower()
        
        # Match English words with hyphens, underscores, numbers
        normalized = re.sub(r'\b[A-Za-z][\w\-_]*[A-Za-z0-9]\b|\b[A-Za-z]+\b', 
                          normalize_english, normalized)
        
        return normalized


class PkusegTokenizer:
    """pkuseg-based tokenizer"""
    
    def __init__(self, model_name: str = "default"):
        """Initialize pkuseg tokenizer with domain-general model"""
        try:
            self.seg = pkuseg.pkuseg(model_name=model_name)
            print(f"✅ Initialized pkuseg with model: {model_name}")
        except Exception as e:
            raise RuntimeError(f"Failed to initialize pkuseg: {e}")
    
    def tokenize(self, text: str) -> List[str]:
        """Tokenize using pkuseg"""
        if not text:
            return []
        return self.seg.cut(text)


class JiebaTokenizer:
    """jieba-based tokenizer as fallback"""
    
    def __init__(self, userdict_path: Optional[str] = None):
        """Initialize jieba tokenizer"""
        if userdict_path and os.path.exists(userdict_path):
            jieba.load_userdict(userdict_path)
            print(f"✅ Loaded jieba user dictionary: {userdict_path}")
        print("✅ Initialized jieba tokenizer")
    
    def tokenize(self, text: str) -> List[str]:
        """Tokenize using jieba"""
        if not text:
            return []
        return list(jieba.cut(text))


class MixedLanguageTokenizer:
    """Mixed Chinese-English tokenizer with pluggable backends"""
    
    def __init__(self, 
                 tokenizer_type: str = "auto",
                 userdict_path: Optional[str] = None,
                 stopwords_zh_path: Optional[str] = None,
                 stopwords_en_path: Optional[str] = None,
                 allow_singletons_path: Optional[str] = None):
        """
        Initialize mixed language tokenizer
        
        Args:
            tokenizer_type: "pkuseg", "jieba", or "auto" (try pkuseg first)
            userdict_path: Path to user dictionary
            stopwords_zh_path: Path to Chinese stopwords
            stopwords_en_path: Path to English stopwords  
            allow_singletons_path: Path to allowed Chinese single characters
        """
        self.normalizer = ChineseNormalizer()
        
        # Initialize tokenizer based on availability and preference
        if tokenizer_type == "auto":
            if PKUSEG_AVAILABLE:
                self.tokenizer = PkusegTokenizer()
                self.tokenizer_name = "pkuseg"
            else:
                self.tokenizer = JiebaTokenizer(userdict_path)
                self.tokenizer_name = "jieba"
        elif tokenizer_type == "pkuseg":
            if PKUSEG_AVAILABLE:
                self.tokenizer = PkusegTokenizer()
                self.tokenizer_name = "pkuseg"
            else:
                raise RuntimeError("pkuseg not available")
        elif tokenizer_type == "jieba":
            self.tokenizer = JiebaTokenizer(userdict_path)
            self.tokenizer_name = "jieba"
        else:
            raise ValueError(f"Unknown tokenizer type: {tokenizer_type}")
        
        # Load stopwords and allowed singletons
        self.stopwords_zh = self._load_stopwords(stopwords_zh_path) if stopwords_zh_path else set()
        self.stopwords_en = self._load_stopwords(stopwords_en_path) if stopwords_en_path else set()
        self.allowed_singletons = self._load_stopwords(allow_singletons_path) if allow_singletons_path else set()
        
        print(f"✅ Tokenizer ready: {self.tokenizer_name}")
        print(f"   - Chinese stopwords: {len(self.stopwords_zh)}")
        print(f"   - English stopwords: {len(self.stopwords_en)}")
        print(f"   - Allowed singletons: {len(self.allowed_singletons)}")
    
    def _load_stopwords(self, path: str) -> Set[str]:
        """Load stopwords from file"""
        if not path or not os.path.exists(path):
            return set()
        
        try:
            with open(path, 'r', encoding='utf-8') as f:
                words = {line.strip() for line in f if line.strip() and not line.startswith('#')}
            return words
        except Exception as e:
            warnings.warn(f"Failed to load stopwords from {path}: {e}")
            return set()
    
    def _is_chinese_char(self, char: str) -> bool:
        """Check if character is Chinese"""
        return '\u4e00' <= char <= '\u9fff'
    
    def _is_english_word(self, word: str) -> bool:
        """Check if word is English (including technical terms with numbers/hyphens)"""
        return bool(re.match(r'^[a-zA-Z][\w\-_]*$', word))
    
    def _filter_token(self, token: str) -> bool:
        """
        Filter tokens based on linguistic rules
        - Remove pure punctuation
        - Remove stopwords  
        - Keep meaningful single Chinese characters
        - Keep technical English terms
        """
        if not token or not token.strip():
            return False
        
        token = token.strip()
        
        # Remove pure punctuation/symbols
        if re.match(r'^[^\w\u4e00-\u9fff]+$', token):
            return False
        
        # Check stopwords
        if token.lower() in self.stopwords_en:
            return False
        if token in self.stopwords_zh:
            return False
        
        # Handle single characters
        if len(token) == 1:
            if self._is_chinese_char(token):
                # Allow meaningful single Chinese characters
                return token in self.allowed_singletons
            else:
                # Filter out single English characters/numbers
                return False
        
        # Keep technical terms and valid words
        return True
    
    def tokenize(self, text: str) -> List[str]:
        """
        Tokenize mixed Chinese-English text
        
        Args:
            text: Input text
            
        Returns:
            List[str]: Filtered tokens
        """
        if not text:
            return []
        
        # Normalize text
        normalized_text = self.normalizer.normalize(text)
        
        # Primary tokenization using backend
        tokens = self.tokenizer.tokenize(normalized_text)
        
        # Post-process: extract English terms separately for better handling
        final_tokens = []
        for token in tokens:
            if not token or not token.strip():
                continue
                
            # For mixed Chinese-English tokens, try to separate them
            if self._contains_both_languages(token):
                separated = self._separate_mixed_token(token)
                final_tokens.extend(separated)
            else:
                final_tokens.append(token)
        
        # Filter tokens
        filtered_tokens = [token for token in final_tokens if self._filter_token(token)]
        
        return filtered_tokens
    
    def _contains_both_languages(self, token: str) -> bool:
        """Check if token contains both Chinese and English"""
        has_chinese = any('\u4e00' <= char <= '\u9fff' for char in token)
        has_english = any(char.isalpha() for char in token)
        return has_chinese and has_english
    
    def _separate_mixed_token(self, token: str) -> List[str]:
        """Separate mixed Chinese-English tokens"""
        # Use regex to split by language boundaries
        parts = re.findall(r'[\u4e00-\u9fff]+|[a-zA-Z][\w\-_]*|\d+', token)
        return [part for part in parts if part]
    
    def get_tokenizer_info(self) -> dict:
        """Get tokenizer information for reporting"""
        return {
            'name': self.tokenizer_name,
            'chinese_stopwords': len(self.stopwords_zh),
            'english_stopwords': len(self.stopwords_en),
            'allowed_singletons': len(self.allowed_singletons),
            'pkuseg_available': PKUSEG_AVAILABLE
        }