"""
Text processing pipeline for Chinese text analysis.
Supports jieba tokenization, n-gram extraction, and various text preprocessing options.
"""

import re
import logging
from typing import Dict, List, Literal, Optional, Set, Tuple
from collections import Counter
import pandas as pd

import jieba
import jieba.posseg as pseg
from opencc import OpenCC
from unidecode import unidecode

# Configure logging
logger = logging.getLogger(__name__)

# Initialize OpenCC for traditional/simplified conversion
try:
    t2s_converter = OpenCC('t2s')  # Traditional to Simplified
    s2t_converter = OpenCC('s2t')  # Simplified to Traditional
except Exception as e:
    logger.warning(f"OpenCC初始化失败: {e}")
    t2s_converter = None
    s2t_converter = None


def clean_text(text: str) -> str:
    """
    Clean text by removing URLs, emails, excessive whitespace, and unwanted characters.
    
    Args:
        text: Input text string
        
    Returns:
        Cleaned text string
    """
    if not text:
        return ""
    
    # Remove URLs
    text = re.sub(r'https?://[^\s]+', '', text)
    text = re.sub(r'www\.[^\s]+', '', text)
    
    # Remove email addresses
    text = re.sub(r'\S+@\S+', '', text)
    
    # Remove excessive punctuation and special characters (keep Chinese punctuation)
    text = re.sub(r'[^\w\s\u4e00-\u9fff\u3000-\u303f\uff00-\uffef]', ' ', text)
    
    # Remove excessive whitespace
    text = re.sub(r'\s+', ' ', text)
    
    # Remove standalone numbers (but keep numbers within words)
    text = re.sub(r'\b\d+\b', ' ', text)
    
    return text.strip()


def apply_traditional_simplified_conversion(text: str, unify_trad_simp: bool) -> str:
    """
    Convert traditional Chinese to simplified if requested.
    
    Args:
        text: Input text
        unify_trad_simp: Whether to convert traditional to simplified
        
    Returns:
        Converted text
    """
    if not unify_trad_simp or not t2s_converter:
        return text
    
    try:
        return t2s_converter.convert(text)
    except Exception as e:
        logger.warning(f"繁简转换失败: {e}")
        return text


def jieba_tokenize(text: str, mode: Literal["jieba_exact", "jieba_search"]) -> List[str]:
    """
    Tokenize text using jieba.
    
    Args:
        text: Input text
        mode: Jieba tokenization mode
        
    Returns:
        List of tokens
    """
    if mode == "jieba_exact":
        return list(jieba.cut(text, cut_all=False))
    elif mode == "jieba_search":
        return list(jieba.cut_for_search(text))
    else:
        raise ValueError(f"Unsupported jieba mode: {mode}")


def ngram_tokenize(text: str, n: int) -> List[str]:
    """
    Generate n-grams from text.
    For Chinese: character-based n-grams
    For English/other: word-based n-grams
    
    Args:
        text: Input text
        n: N-gram size (1-6)
        
    Returns:
        List of n-grams
    """
    if n < 1 or n > 6:
        raise ValueError("n必须在1-6之间")
    
    # Clean whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    
    if not text:
        return []
    
    # Check if text is primarily Chinese
    chinese_chars = len(re.findall(r'[\u4e00-\u9fff]', text))
    total_chars = len(re.sub(r'\s', '', text))
    
    if chinese_chars / max(total_chars, 1) > 0.5:
        # Chinese: character-based n-grams
        chars = [c for c in text if c != ' ']
        if len(chars) < n:
            return []
        return [''.join(chars[i:i+n]) for i in range(len(chars) - n + 1)]
    else:
        # English/other: word-based n-grams
        words = text.split()
        if len(words) < n:
            return []
        return [' '.join(words[i:i+n]) for i in range(len(words) - n + 1)]


def filter_tokens(tokens: List[str], 
                 stopwords: Set[str], 
                 keep_whitelist: Set[str]) -> List[str]:
    """
    Filter tokens using stopwords and whitelist.
    Whitelist takes priority over stopwords.
    
    Args:
        tokens: List of tokens
        stopwords: Set of stopwords to remove
        keep_whitelist: Set of words to always keep
        
    Returns:
        Filtered tokens
    """
    filtered = []
    for token in tokens:
        token_clean = token.strip()
        if not token_clean:
            continue
            
        # Whitelist takes priority
        if token_clean in keep_whitelist:
            filtered.append(token_clean)
        elif token_clean not in stopwords:
            # Additional filtering: remove very short tokens (< 2 chars for Chinese)
            if len(token_clean) >= 2 or not re.search(r'[\u4e00-\u9fff]', token_clean):
                filtered.append(token_clean)
    
    return filtered


def apply_synonym_mapping(tokens: List[str], synonyms: Dict[str, str]) -> List[str]:
    """
    Apply synonym mapping to tokens.
    
    Args:
        tokens: List of tokens
        synonyms: Dictionary mapping synonyms to canonical forms
        
    Returns:
        Tokens with synonyms replaced
    """
    if not synonyms:
        return tokens
    
    mapped = []
    for token in tokens:
        # Check exact match first
        if token in synonyms:
            mapped.append(synonyms[token])
        else:
            # Check case-insensitive match
            token_lower = token.lower()
            synonym_found = False
            for synonym, canonical in synonyms.items():
                if synonym.lower() == token_lower:
                    mapped.append(canonical)
                    synonym_found = True
                    break
            if not synonym_found:
                mapped.append(token)
    
    return mapped


def tokenize(records: List[Dict], 
            mode: Literal["jieba_exact", "jieba_search", "ngram"],
            n: int = 2,
            stopwords: Set[str] = None,
            keep_whitelist: Set[str] = None,
            synonyms: Dict[str, str] = None,
            unify_trad_simp: bool = False) -> List[str]:
    """
    Tokenize corpus using specified method and filters.
    
    Args:
        records: List of record dictionaries with 'content' field
        mode: Tokenization mode
        n: N-gram size (for ngram mode)
        stopwords: Set of stopwords to remove
        keep_whitelist: Set of words to always keep
        synonyms: Dictionary for synonym mapping
        unify_trad_simp: Whether to convert traditional to simplified Chinese
        
    Returns:
        List of all tokens from the corpus
    """
    if stopwords is None:
        stopwords = set()
    if keep_whitelist is None:
        keep_whitelist = set()
    if synonyms is None:
        synonyms = {}
    
    all_tokens = []
    
    for record in records:
        content = record.get('content', '')
        if not content:
            continue
        
        # Clean text
        text = clean_text(content)
        if not text:
            continue
        
        # Apply traditional/simplified conversion
        text = apply_traditional_simplified_conversion(text, unify_trad_simp)
        
        # Tokenize based on mode
        if mode in ["jieba_exact", "jieba_search"]:
            tokens = jieba_tokenize(text, mode)
        elif mode == "ngram":
            tokens = ngram_tokenize(text, n)
        else:
            raise ValueError(f"不支持的分词模式: {mode}")
        
        # Filter tokens
        tokens = filter_tokens(tokens, stopwords, keep_whitelist)
        
        # Apply synonym mapping
        tokens = apply_synonym_mapping(tokens, synonyms)
        
        all_tokens.extend(tokens)
    
    return all_tokens


def compute_frequencies(tokens: List[str]) -> pd.DataFrame:
    """
    Compute token frequencies and proportions.
    
    Args:
        tokens: List of tokens
        
    Returns:
        DataFrame with columns: term, freq, prop
    """
    if not tokens:
        return pd.DataFrame(columns=['term', 'freq', 'prop'])
    
    # Count frequencies
    counter = Counter(tokens)
    total_tokens = len(tokens)
    
    # Create dataframe
    freq_data = []
    for term, freq in counter.most_common():
        prop = freq / total_tokens
        freq_data.append({
            'term': term,
            'freq': freq,
            'prop': round(prop, 6)
        })
    
    return pd.DataFrame(freq_data)


def kwic_index(records: List[Dict], 
               term: str, 
               window: int = 30) -> List[Dict[str, str]]:
    """
    Generate KWIC (Key Word In Context) index for a term.
    
    Args:
        records: List of record dictionaries
        term: Term to search for
        window: Context window size (characters on each side)
        
    Returns:
        List of KWIC entries with context, title, and URL
    """
    kwic_entries = []
    
    for record in records:
        content = record.get('content', '')
        title = record.get('title', '未知标题')
        url = record.get('url', '')
        
        if not content:
            continue
        
        # Find all occurrences of the term (case-insensitive)
        content_lower = content.lower()
        term_lower = term.lower()
        
        start = 0
        while True:
            pos = content_lower.find(term_lower, start)
            if pos == -1:
                break
            
            # Extract context
            left_start = max(0, pos - window)
            right_end = min(len(content), pos + len(term) + window)
            
            left_context = content[left_start:pos]
            keyword = content[pos:pos + len(term)]
            right_context = content[pos + len(term):right_end]
            
            # Clean context (remove excessive whitespace)
            left_context = re.sub(r'\s+', ' ', left_context).strip()
            right_context = re.sub(r'\s+', ' ', right_context).strip()
            
            kwic_entries.append({
                'left': left_context,
                'keyword': keyword,
                'right': right_context,
                'title': title,
                'url': url
            })
            
            start = pos + 1
    
    return kwic_entries


def load_stopwords_from_file(file_path: str) -> Set[str]:
    """
    Load stopwords from text file.
    
    Args:
        file_path: Path to stopwords file
        
    Returns:
        Set of stopwords
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            stopwords = set()
            for line in f:
                word = line.strip()
                if word and not word.startswith('#'):
                    stopwords.add(word)
        return stopwords
    except Exception as e:
        logger.warning(f"加载停用词文件失败 {file_path}: {e}")
        return set()


def load_synonyms_from_csv(file_path: str) -> Dict[str, str]:
    """
    Load synonym mapping from CSV file.
    Expected format: from,to
    
    Args:
        file_path: Path to CSV file
        
    Returns:
        Dictionary mapping synonyms to canonical forms
    """
    try:
        df = pd.read_csv(file_path)
        if 'from' not in df.columns or 'to' not in df.columns:
            logger.warning(f"同义词文件格式错误，需要包含'from'和'to'列: {file_path}")
            return {}
        
        synonyms = {}
        for _, row in df.iterrows():
            from_term = str(row['from']).strip()
            to_term = str(row['to']).strip()
            if from_term and to_term:
                synonyms[from_term] = to_term
        
        return synonyms
    except Exception as e:
        logger.warning(f"加载同义词文件失败 {file_path}: {e}")
        return {}


if __name__ == "__main__":
    # Test tokenization
    test_records = [
        {'content': '这是一个测试文本，包含中文和English混合内容。'},
        {'content': '人工智能和机器学习是未来科技发展的重要方向。'}
    ]
    
    print("=== Jieba精确模式测试 ===")
    tokens = tokenize(test_records, mode="jieba_exact")
    print(f"Tokens: {tokens[:10]}")
    
    print("\n=== N-gram测试 (n=2) ===")
    tokens = tokenize(test_records, mode="ngram", n=2)
    print(f"N-grams: {tokens[:10]}")
    
    print("\n=== 频次统计测试 ===")
    freq_df = compute_frequencies(tokens)
    print(freq_df.head())
    
    print("\n=== KWIC测试 ===")
    kwic_results = kwic_index(test_records, "测试")
    print(f"KWIC entries: {len(kwic_results)}")
    if kwic_results:
        print(f"示例: {kwic_results[0]}")