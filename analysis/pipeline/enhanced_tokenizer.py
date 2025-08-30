#!/usr/bin/env python3
"""
Enhanced tokenization module with pkuseg→jieba fallback and user dictionary hot-loading
Provides configuration interface for TOKENIZER_TYPE and EXTRA_USER_DICTS
"""

import os
from typing import List, Dict, Optional, Tuple
from pathlib import Path

# Import the core implementation
from tokenizer import MixedLanguageTokenizer, TokenInfo

# Configuration constants
TOKENIZER_TYPE = {"auto", "pkuseg", "jieba"}
EXTRA_USER_DICTS = []  # List of user dictionary paths

# Default configuration
DEFAULT_DATA_DIR = Path(__file__).parent.parent / "data"
DEFAULT_USER_DICTS = [
    str(DEFAULT_DATA_DIR / "user_dict.zh.txt"),
    str(DEFAULT_DATA_DIR / "tech_terms.txt")
]
DEFAULT_STOPWORDS_ZH = str(DEFAULT_DATA_DIR / "stopwords.zh.txt")
DEFAULT_STOPWORDS_EN = str(DEFAULT_DATA_DIR / "stopwords.en.txt")
DEFAULT_SINGLETONS = str(DEFAULT_DATA_DIR / "allow_singletons.zh.txt")


class EnhancedTokenizer:
    """
    Enhanced tokenizer with configuration support for TOKENIZER_TYPE and EXTRA_USER_DICTS
    """
    
    def __init__(self, 
                 tokenizer_type: str = "auto",
                 extra_user_dicts: Optional[List[str]] = None,
                 use_default_dicts: bool = True):
        """
        Initialize tokenizer with configuration
        
        Args:
            tokenizer_type: One of "auto", "pkuseg", "jieba"
            extra_user_dicts: Additional user dictionary paths
            use_default_dicts: Whether to include default user dictionaries
        """
        if tokenizer_type not in TOKENIZER_TYPE:
            raise ValueError(f"tokenizer_type must be one of {TOKENIZER_TYPE}")
        
        # Combine default and extra user dictionaries
        user_dicts = []
        if use_default_dicts:
            user_dicts.extend(DEFAULT_USER_DICTS)
        if extra_user_dicts:
            user_dicts.extend(extra_user_dicts)
        if EXTRA_USER_DICTS:
            user_dicts.extend(EXTRA_USER_DICTS)
        
        # Initialize the core tokenizer
        self.tokenizer = MixedLanguageTokenizer(
            tokenizer_type=tokenizer_type,
            extra_user_dicts=user_dicts,
            stopwords_zh_path=DEFAULT_STOPWORDS_ZH,
            stopwords_en_path=DEFAULT_STOPWORDS_EN,
            allow_singletons_path=DEFAULT_SINGLETONS
        )
    
    def tokenize(self, text: str) -> List[str]:
        """
        Tokenize text with normalization and filtering
        
        Args:
            text: Input text
            
        Returns:
            List of tokens
        """
        return self.tokenizer.tokenize(text)
    
    def inspect(self, text: str, topk: int = 50) -> Dict[str, List[TokenInfo]]:
        """
        Inspect tokenization process for debugging and auditing
        
        Args:
            text: Input text
            topk: Top K tokens to return for each category
            
        Returns:
            Dict with 'kept' and 'filtered' token information containing:
            - token: str - the token text
            - char_span: Tuple[int, int] - (start, end) character positions
            - source: str - 'pkuseg', 'jieba', 'mixed_separation', 'normalization'
            - kept_reason: str - why this token was kept or filtered
        """
        return self.tokenizer.inspect(text, topk)
    
    def get_info(self) -> dict:
        """Get tokenizer information for reporting"""
        return self.tokenizer.get_tokenizer_info()


# Convenience functions for direct usage
def create_tokenizer(tokenizer_type: str = "auto", 
                    extra_user_dicts: Optional[List[str]] = None) -> EnhancedTokenizer:
    """
    Create a tokenizer with the specified configuration
    
    Args:
        tokenizer_type: One of "auto", "pkuseg", "jieba"
        extra_user_dicts: Additional user dictionary paths
        
    Returns:
        Configured tokenizer instance
    """
    return EnhancedTokenizer(tokenizer_type, extra_user_dicts)


def tokenize_text(text: str, 
                 tokenizer_type: str = "auto",
                 extra_user_dicts: Optional[List[str]] = None) -> List[str]:
    """
    Convenience function to tokenize text directly
    
    Args:
        text: Input text
        tokenizer_type: One of "auto", "pkuseg", "jieba"
        extra_user_dicts: Additional user dictionary paths
        
    Returns:
        List of tokens
    """
    tokenizer = create_tokenizer(tokenizer_type, extra_user_dicts)
    return tokenizer.tokenize(text)


def inspect_tokenization(text: str,
                        tokenizer_type: str = "auto", 
                        extra_user_dicts: Optional[List[str]] = None,
                        topk: int = 50) -> Dict[str, List[TokenInfo]]:
    """
    Convenience function to inspect tokenization process
    
    Args:
        text: Input text
        tokenizer_type: One of "auto", "pkuseg", "jieba"  
        extra_user_dicts: Additional user dictionary paths
        topk: Top K tokens to return for each category
        
    Returns:
        Dict with 'kept' and 'filtered' token information
    """
    tokenizer = create_tokenizer(tokenizer_type, extra_user_dicts)
    return tokenizer.inspect(text, topk)


# Module-level configuration
def set_global_user_dicts(dicts: List[str]):
    """Set global user dictionaries that will be used by all tokenizers"""
    global EXTRA_USER_DICTS
    EXTRA_USER_DICTS.clear()
    EXTRA_USER_DICTS.extend(dicts)


def get_global_user_dicts() -> List[str]:
    """Get current global user dictionaries"""
    return EXTRA_USER_DICTS.copy()


def add_global_user_dict(dict_path: str):
    """Add a user dictionary to global configuration"""
    global EXTRA_USER_DICTS
    if dict_path not in EXTRA_USER_DICTS:
        EXTRA_USER_DICTS.append(dict_path)


# Export main classes and functions
__all__ = [
    'EnhancedTokenizer',
    'TokenInfo', 
    'TOKENIZER_TYPE',
    'EXTRA_USER_DICTS',
    'create_tokenizer',
    'tokenize_text',
    'inspect_tokenization',
    'set_global_user_dicts',
    'get_global_user_dicts', 
    'add_global_user_dict'
]