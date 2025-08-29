#!/usr/bin/env python3
"""
Enhanced tokenization and normalization core
- pkuseg primary with automatic fallback to jieba
- User dictionary hot-loading support
- Mixed language (Chinese + English) recognition and preservation  
- Technical terms (AIGC_2025, ResNet50, etc.) preservation
- Camel/snake case preservation
- Audit trail for token inspection
"""

import os
import re
from typing import List, Set, Optional, Protocol, Dict, Tuple, NamedTuple
import warnings
import unicodedata
from pathlib import Path

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


class TokenInfo(NamedTuple):
    """Token information for inspection and auditing"""
    token: str
    char_span: Tuple[int, int]  # (start, end) character positions
    source: str  # 'pkuseg', 'jieba', 'mixed_separation', 'normalization'
    kept_reason: str  # why this token was kept or filtered


class TokenizerProtocol(Protocol):
    """Protocol for pluggable tokenizers"""
    
    def tokenize(self, text: str) -> List[str]:
        """Tokenize text into list of tokens"""
        ...


class ChineseNormalizer:
    """Enhanced text normalizer for Chinese and mixed-language content"""
    
    @staticmethod
    def normalize(text: str, preserve_case_patterns: bool = True) -> str:
        """
        Normalize text with Chinese-first approach
        - Convert full-width to half-width
        - Convert English to lowercase (with pattern preservation)
        - Preserve technical terms: AIGC_2025, ResNet50, camelCase, snake_case
        - Merge excessive whitespace and punctuation
        """
        if not text:
            return ""
            
        # Convert full-width to half-width using NFKC
        normalized = unicodedata.normalize('NFKC', text)
        
        # Replace full-width punctuation with half-width
        replacements = {
            '，': ',', '。': '.', '？': '?', '！': '!', 
            '：': ':', '；': ';', '"': '"', '"': '"',
            ''': "'", ''': "'", '（': '(', '）': ')',
            '【': '[', '】': ']', '《': '<', '》': '>',
            '、': ',', '　': ' ',  # Full-width space to half-width
            '＋': '+', '－': '-', '＝': '=', '％': '%',
            '＠': '@', '＃': '#', '＄': '$', '＆': '&'
        }
        
        for full, half in replacements.items():
            normalized = normalized.replace(full, half)
        
        # Preserve technical patterns before lowercasing
        if preserve_case_patterns:
            # Store patterns that should preserve case
            preserved_patterns = []
            
            # Match technical terms: AIGC_2025, ResNet50, ChatGPT-4o-mini, IL-6/STAT3, camelCase, snake_case
            tech_patterns = [
                r'\b[A-Z]{2,}(?:[_\-/][A-Za-z0-9]+)*(?:[_\-]?\d+[A-Za-z]*)*\b',  # AIGC_2025, IL-6/STAT3
                r'\b[A-Z][a-z]+(?:[A-Z][a-z]*)+(?:\d+[A-Za-z]*)*\b',  # camelCase, ResNet50
                r'\b[a-z]+(?:[A-Z][a-z]*)+\b',  # camelCase starting with lowercase
                r'\b[a-z]+(?:_[a-z0-9]+)+\b',  # snake_case
                r'\b[A-Za-z]+(?:[_\-]\d+[A-Za-z]*)+\b',  # mixed patterns
                r'\b[A-Za-z]+\d+(?:[A-Za-z]+\d*)*\b',  # alphanumeric like ResNet50
                r'\b[A-Z]+/[A-Z]+\d*\b',  # IL-6/STAT3 style
                r'\b[A-Z][a-z]+(?:-\d+[a-z]*)+(?:-[a-z]+)*\b'  # ChatGPT-4o-mini style
            ]
            
            for pattern in tech_patterns:
                for match in re.finditer(pattern, normalized):
                    preserved_patterns.append((match.span(), match.group()))
            
            # Sort by position to avoid conflicts during replacement
            preserved_patterns.sort(key=lambda x: x[0])
            
            # Convert to lowercase
            normalized_lower = normalized.lower()
            
            # Restore preserved patterns (work backwards to maintain positions)
            for (start, end), original in reversed(preserved_patterns):
                before = normalized_lower[:start]
                after = normalized_lower[end:]
                normalized_lower = before + original + after
            
            normalized = normalized_lower
        else:
            # Simple lowercase without preservation for regular English words
            def lowercase_english(match):
                return match.group(0).lower()
            
            normalized = re.sub(r'\b[a-zA-Z]+\b', lowercase_english, normalized)
        
        # Merge excessive whitespace and punctuation
        normalized = re.sub(r'\s+', ' ', normalized)  # Multiple spaces -> single space
        normalized = re.sub(r'[,，]{2,}', ',', normalized)  # Multiple commas
        normalized = re.sub(r'[.。]{2,}', '.', normalized)  # Multiple periods
        normalized = normalized.strip()
        
        return normalized


class PkusegTokenizer:
    """Enhanced pkuseg-based tokenizer with domain support"""
    
    def __init__(self, model_name: str = "default", user_dict_paths: Optional[List[str]] = None):
        """
        Initialize pkuseg tokenizer
        
        Args:
            model_name: pkuseg model ('default', 'news', 'web', 'medicine', 'tourism')
            user_dict_paths: List of user dictionary paths for hot-loading
        """
        try:
            # Initialize with domain-general model first
            self.seg = pkuseg.pkuseg(model_name=model_name)
            self.model_name = model_name
            print(f"✅ Initialized pkuseg with model: {model_name}")
            
            # Load user dictionaries if provided
            self.user_dict_paths = user_dict_paths or []
            self._load_user_dicts()
            
        except Exception as e:
            raise RuntimeError(f"Failed to initialize pkuseg: {e}")
    
    def _load_user_dicts(self):
        """Load user dictionaries for pkuseg (approximation via preprocessing)"""
        self.user_terms = set()
        loaded_count = 0
        
        for dict_path in self.user_dict_paths:
            if dict_path and os.path.exists(dict_path):
                try:
                    with open(dict_path, 'r', encoding='utf-8') as f:
                        for line in f:
                            line = line.strip()
                            if line and not line.startswith('#'):
                                # Extract term (first column)
                                term = line.split()[0] if line.split() else line
                                if term:
                                    self.user_terms.add(term)
                                    loaded_count += 1
                    print(f"✅ Loaded {len([term for term in self.user_terms if any(term in dict_path for dict_path in [dict_path])])} terms from {dict_path}")
                except Exception as e:
                    warnings.warn(f"Failed to load user dict {dict_path}: {e}")
        
        if loaded_count > 0:
            print(f"✅ Total user dictionary terms loaded: {len(self.user_terms)}")
    
    def tokenize(self, text: str) -> List[str]:
        """Tokenize using pkuseg with user dictionary awareness"""
        if not text:
            return []
        
        # Preprocess to protect user dictionary terms
        protected_text = self._protect_user_terms(text)
        
        # Tokenize with pkuseg
        tokens = self.seg.cut(protected_text)
        
        # Post-process to restore protected terms
        tokens = self._restore_user_terms(tokens)
        
        return tokens
    
    def _protect_user_terms(self, text: str) -> str:
        """Protect user dictionary terms during tokenization"""
        protected = text
        for term in sorted(self.user_terms, key=len, reverse=True):
            # Replace with protected placeholder
            protected = protected.replace(term, f"__TERM_{term}__")
        return protected
    
    def _restore_user_terms(self, tokens: List[str]) -> List[str]:
        """Restore protected user terms"""
        restored = []
        for token in tokens:
            if token.startswith("__TERM_") and token.endswith("__"):
                # Extract original term
                original = token[7:-2]  # Remove __TERM_ and __
                restored.append(original)
            else:
                restored.append(token)
        return restored


class JiebaTokenizer:
    """Enhanced jieba-based tokenizer as fallback"""
    
    def __init__(self, user_dict_paths: Optional[List[str]] = None):
        """
        Initialize jieba tokenizer with user dictionary support
        
        Args:
            user_dict_paths: List of user dictionary paths for hot-loading
        """
        self.user_dict_paths = user_dict_paths or []
        self._load_user_dicts()
        print("✅ Initialized jieba tokenizer")
    
    def _load_user_dicts(self):
        """Load user dictionaries for jieba"""
        loaded_count = 0
        
        for dict_path in self.user_dict_paths:
            if dict_path and os.path.exists(dict_path):
                try:
                    # Load original dictionary
                    jieba.load_userdict(dict_path)
                    
                    # Also load normalized versions for better matching
                    normalized_terms = []
                    with open(dict_path, 'r', encoding='utf-8') as f:
                        for line in f:
                            line = line.strip()
                            if line and not line.startswith('#'):
                                # Extract term (first column)
                                parts = line.split()
                                if parts:
                                    original_term = parts[0]
                                    # Create normalized version (simplified normalization)
                                    normalized_term = original_term.lower()
                                    
                                    # Add normalized version if different
                                    if normalized_term != original_term and normalized_term not in normalized_terms:
                                        normalized_terms.append(normalized_term)
                                        # Add to jieba with frequency
                                        freq = int(parts[1]) if len(parts) > 1 and parts[1].isdigit() else 100
                                        jieba.add_word(normalized_term, freq)
                    
                    # Count loaded terms for reporting
                    with open(dict_path, 'r', encoding='utf-8') as f:
                        count = sum(1 for line in f if line.strip() and not line.startswith('#'))
                    loaded_count += count + len(normalized_terms)  # original + normalized
                    print(f"✅ Loaded jieba user dictionary: {dict_path} ({count} terms + {len(normalized_terms)} normalized variants)")
                except Exception as e:
                    warnings.warn(f"Failed to load user dict {dict_path}: {e}")
        
        if loaded_count > 0:
            print(f"✅ Total user dictionary terms loaded: {loaded_count}")
    
    def tokenize(self, text: str) -> List[str]:
        """Tokenize using jieba"""
        if not text:
            return []
        return list(jieba.cut(text))


class MixedLanguageTokenizer:
    """Enhanced mixed Chinese-English tokenizer with hot-loading and audit trail"""
    
    def __init__(self, 
                 tokenizer_type: str = "auto",
                 extra_user_dicts: Optional[List[str]] = None,
                 stopwords_zh_path: Optional[str] = None,
                 stopwords_en_path: Optional[str] = None,
                 allow_singletons_path: Optional[str] = None):
        """
        Initialize enhanced mixed language tokenizer
        
        Args:
            tokenizer_type: "pkuseg", "jieba", or "auto" (try pkuseg first)
            extra_user_dicts: List of user dictionary paths for hot-loading
            stopwords_zh_path: Path to Chinese stopwords
            stopwords_en_path: Path to English stopwords  
            allow_singletons_path: Path to allowed Chinese single characters
        """
        self.normalizer = ChineseNormalizer()
        self.extra_user_dicts = extra_user_dicts or []
        
        # Initialize tokenizer based on availability and preference
        if tokenizer_type == "auto":
            if PKUSEG_AVAILABLE:
                self.tokenizer = PkusegTokenizer(user_dict_paths=self.extra_user_dicts)
                self.tokenizer_name = "pkuseg"
            else:
                self.tokenizer = JiebaTokenizer(user_dict_paths=self.extra_user_dicts)
                self.tokenizer_name = "jieba"
        elif tokenizer_type == "pkuseg":
            if PKUSEG_AVAILABLE:
                self.tokenizer = PkusegTokenizer(user_dict_paths=self.extra_user_dicts)
                self.tokenizer_name = "pkuseg"
            else:
                print("⚠️  pkuseg not available, falling back to jieba")
                self.tokenizer = JiebaTokenizer(user_dict_paths=self.extra_user_dicts)
                self.tokenizer_name = "jieba (fallback)"
        elif tokenizer_type == "jieba":
            self.tokenizer = JiebaTokenizer(user_dict_paths=self.extra_user_dicts)
            self.tokenizer_name = "jieba"
        else:
            raise ValueError(f"Unknown tokenizer type: {tokenizer_type}")
        
        # Load stopwords and allowed singletons
        self.stopwords_zh = self._load_word_set(stopwords_zh_path) if stopwords_zh_path else set()
        self.stopwords_en = self._load_word_set(stopwords_en_path) if stopwords_en_path else set()
        self.allowed_singletons = self._load_word_set(allow_singletons_path) if allow_singletons_path else set()
        
        print(f"✅ Tokenizer ready: {self.tokenizer_name}")
        print(f"   - Chinese stopwords: {len(self.stopwords_zh)}")
        print(f"   - English stopwords: {len(self.stopwords_en)}")
        print(f"   - Allowed singletons: {len(self.allowed_singletons)}")
        print(f"   - User dictionaries: {len(self.extra_user_dicts)}")
    
    def _load_word_set(self, path: str) -> Set[str]:
        """Load word set from file"""
        if not path or not os.path.exists(path):
            return set()
        
        try:
            with open(path, 'r', encoding='utf-8') as f:
                words = {line.strip() for line in f if line.strip() and not line.startswith('#')}
            return words
        except Exception as e:
            warnings.warn(f"Failed to load word set from {path}: {e}")
            return set()
    
    def _is_chinese_char(self, char: str) -> bool:
        """Check if character is Chinese"""
        return '\u4e00' <= char <= '\u9fff'
    
    def _is_technical_term(self, word: str) -> bool:
        """Check if word is a technical term that should be preserved"""
        # Technical patterns: AIGC_2025, ResNet50, ChatGPT-4o-mini, IL-6/STAT3, camelCase, snake_case
        patterns = [
            r'^[A-Z]{2,}(?:[_\-][A-Za-z0-9]+)*(?:[_\-]?\d+[A-Za-z]*)*$',  # AIGC_2025, IL-6
            r'^[A-Z][a-z]+(?:[A-Z][a-z]*)*(?:\d+[A-Za-z]*)*$',  # camelCase, ResNet50
            r'^[a-z]+(?:_[a-z0-9]+)+$',  # snake_case
            r'^[A-Za-z]+(?:[_\-]\d+[A-Za-z]*)+$',  # mixed patterns
            r'^[A-Za-z]+\d+(?:[A-Za-z]+\d*)*$',  # alphanumeric
            r'^[A-Z]+/[A-Z]+\d*$'  # IL-6/STAT3 style
        ]
        
        return any(re.match(pattern, word) for pattern in patterns)
    
    def _filter_token(self, token: str) -> Tuple[bool, str]:
        """
        Filter tokens based on linguistic rules
        
        Returns:
            (keep: bool, reason: str)
        """
        if not token or not token.strip():
            return False, "empty_or_whitespace"
        
        token = token.strip()
        
        # Remove pure punctuation/symbols
        if re.match(r'^[^\w\u4e00-\u9fff]+$', token):
            return False, "pure_punctuation"
        
        # Preserve technical terms regardless of other rules
        if self._is_technical_term(token):
            return True, "technical_term"
        
        # Check stopwords
        if token.lower() in self.stopwords_en:
            return False, "english_stopword"
        if token in self.stopwords_zh:
            return False, "chinese_stopword"
        
        # Handle single characters
        if len(token) == 1:
            if self._is_chinese_char(token):
                # Allow meaningful single Chinese characters
                if token in self.allowed_singletons:
                    return True, "allowed_chinese_singleton"
                else:
                    return False, "disallowed_chinese_singleton"
            else:
                # Filter out single English characters/numbers unless technical
                return False, "single_char_non_chinese"
        
        # Keep valid words
        return True, "valid_token"
    
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
        
        # Post-process: handle mixed tokens and apply filtering
        final_tokens = []
        for token in tokens:
            if not token or not token.strip():
                continue
                
            # For mixed Chinese-English tokens, try to separate them
            if self._contains_both_languages(token) and not self._is_technical_term(token):
                separated = self._separate_mixed_token(token)
                final_tokens.extend(separated)
            else:
                final_tokens.append(token)
        
        # Filter tokens
        filtered_tokens = []
        for token in final_tokens:
            keep, reason = self._filter_token(token)
            if keep:
                filtered_tokens.append(token)
        
        return filtered_tokens
    
    def inspect(self, text: str, topk: int = 50) -> Dict[str, List[TokenInfo]]:
        """
        Inspect tokenization process for debugging and auditing
        
        Args:
            text: Input text
            topk: Top K tokens to return for each category
            
        Returns:
            Dict with 'kept' and 'filtered' token information
        """
        if not text:
            return {"kept": [], "filtered": []}
        
        # Normalize text
        normalized_text = self.normalizer.normalize(text)
        
        # Primary tokenization
        tokens = self.tokenizer.tokenize(normalized_text)
        
        # Track tokens with positions (approximation)
        kept_tokens = []
        filtered_tokens = []
        
        char_pos = 0
        for token in tokens:
            if not token or not token.strip():
                continue
            
            # Find token position in text (approximation)
            token_start = normalized_text.find(token, char_pos)
            if token_start == -1:
                token_start = char_pos
            token_end = token_start + len(token)
            char_pos = token_end
            
            # Handle mixed tokens
            if self._contains_both_languages(token) and not self._is_technical_term(token):
                separated = self._separate_mixed_token(token)
                sub_pos = token_start
                for sub_token in separated:
                    sub_end = sub_pos + len(sub_token)
                    keep, reason = self._filter_token(sub_token)
                    
                    token_info = TokenInfo(
                        token=sub_token,
                        char_span=(sub_pos, sub_end),
                        source=f"{self.tokenizer_name}_mixed_separation",
                        kept_reason=reason
                    )
                    
                    if keep:
                        kept_tokens.append(token_info)
                    else:
                        filtered_tokens.append(token_info)
                    
                    sub_pos = sub_end
            else:
                keep, reason = self._filter_token(token)
                
                token_info = TokenInfo(
                    token=token,
                    char_span=(token_start, token_end),
                    source=self.tokenizer_name,
                    kept_reason=reason
                )
                
                if keep:
                    kept_tokens.append(token_info)
                else:
                    filtered_tokens.append(token_info)
        
        # Sort by frequency and return top K
        from collections import Counter
        
        kept_counter = Counter(t.token for t in kept_tokens)
        filtered_counter = Counter(t.token for t in filtered_tokens)
        
        top_kept = []
        for token, count in kept_counter.most_common(topk):
            # Find a representative TokenInfo for this token
            representative = next(t for t in kept_tokens if t.token == token)
            top_kept.append(representative)
        
        top_filtered = []
        for token, count in filtered_counter.most_common(topk):
            representative = next(t for t in filtered_tokens if t.token == token)
            top_filtered.append(representative)
        
        return {
            "kept": top_kept,
            "filtered": top_filtered
        }
    
    def _contains_both_languages(self, token: str) -> bool:
        """Check if token contains both Chinese and English"""
        has_chinese = any('\u4e00' <= char <= '\u9fff' for char in token)
        has_english = any(char.isalpha() for char in token)
        return has_chinese and has_english
    
    def _separate_mixed_token(self, token: str) -> List[str]:
        """Separate mixed Chinese-English tokens"""
        # Use regex to split by language boundaries while preserving technical terms
        parts = re.findall(r'[\u4e00-\u9fff]+|[a-zA-Z][\w\-_]*|\d+|[^\w\s]', token)
        return [part for part in parts if part]
    
    def get_tokenizer_info(self) -> dict:
        """Get tokenizer information for reporting"""
        return {
            'name': self.tokenizer_name,
            'backend_available': {
                'pkuseg': PKUSEG_AVAILABLE,
                'jieba': True
            },
            'chinese_stopwords': len(self.stopwords_zh),
            'english_stopwords': len(self.stopwords_en),
            'allowed_singletons': len(self.allowed_singletons),
            'user_dictionaries': len(self.extra_user_dicts),
            'fallback_used': 'fallback' in self.tokenizer_name
        }