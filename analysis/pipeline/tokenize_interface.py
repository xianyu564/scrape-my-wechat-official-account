#!/usr/bin/env python3
"""
Tokenization interface module
Provides TOKENIZER_TYPE and EXTRA_USER_DICTS configuration as required
"""

# Re-export all functionality from enhanced_tokenizer
from enhanced_tokenizer import *

# Configuration constants as required
TOKENIZER_TYPE = {"auto", "pkuseg", "jieba"}
EXTRA_USER_DICTS = []  # List of user dictionary paths

# Convenience aliases for the requirements
def inspect(text: str, topk: int = 50):
    """
    Inspect tokenization process for debugging and auditing
    Returns token/char_span/source/kept_reason information
    """
    tokenizer = EnhancedTokenizer()
    return tokenizer.inspect(text, topk)

# Main tokenizer instance for direct usage
_default_tokenizer = None

def get_default_tokenizer():
    """Get or create default tokenizer instance"""
    global _default_tokenizer
    if _default_tokenizer is None:
        _default_tokenizer = EnhancedTokenizer()
    return _default_tokenizer

def tokenize(text: str):
    """Default tokenize function"""
    return get_default_tokenizer().tokenize(text)