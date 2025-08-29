#!/usr/bin/env python3
"""
Variable-length n-gram builder with collocation filtering (PMI/log-likelihood)
"""

import math
from typing import List, Dict, Tuple, Set, Counter as CounterType
from collections import Counter, defaultdict
import itertools


def build_ngrams(tokens: List[str], 
                max_n: int = 8, 
                min_freq: int = 5, 
                collocation: str = 'pmi',
                pmi_threshold: float = 3.0,
                llr_threshold: float = 10.83) -> Tuple[List[str], Dict[int, int]]:
    """
    Build variable-length n-grams with collocation filtering
    
    Args:
        tokens: List of tokens from corpus
        max_n: Maximum n-gram length (not capped at 1-4)
        min_freq: Minimum frequency threshold  
        collocation: 'pmi' or 'llr' for collocation measure
        pmi_threshold: PMI threshold for collocation filtering
        llr_threshold: Log-likelihood ratio threshold
        
    Returns:
        Tuple[List[str], Dict[int, int]]: (merged_tokens, ngram_counts_by_length)
    """
    if not tokens:
        return [], {}
    
    print(f"🔢 Building n-grams: max_n={max_n}, min_freq={min_freq}, measure={collocation}")
    
    # Step 1: Generate all n-gram candidates
    ngram_candidates = {}
    for n in range(1, max_n + 1):
        ngrams = _extract_ngrams(tokens, n)
        # Filter by frequency first
        frequent_ngrams = {ngram: count for ngram, count in ngrams.items() if count >= min_freq}
        ngram_candidates[n] = frequent_ngrams
        print(f"   - {n}-grams: {len(frequent_ngrams)} candidates (freq >= {min_freq})")
    
    # Step 2: Apply collocation filtering for n > 1
    filtered_ngrams = {}
    filtered_ngrams[1] = ngram_candidates[1]  # Keep all unigrams
    
    for n in range(2, max_n + 1):
        if n not in ngram_candidates or not ngram_candidates[n]:
            filtered_ngrams[n] = {}
            continue
            
        if collocation == 'pmi':
            filtered_ngrams[n] = _filter_by_pmi(
                ngram_candidates[n], 
                ngram_candidates[n-1], 
                ngram_candidates[1],
                threshold=pmi_threshold
            )
        elif collocation == 'llr':
            filtered_ngrams[n] = _filter_by_llr(
                ngram_candidates[n],
                ngram_candidates[n-1],
                ngram_candidates[1], 
                threshold=llr_threshold
            )
        else:
            # No filtering, keep all frequent n-grams
            filtered_ngrams[n] = ngram_candidates[n]
        
        print(f"   - {n}-grams after {collocation}: {len(filtered_ngrams[n])} retained")
    
    # Step 3: Merge n-grams back into token sequence
    merged_tokens = _merge_ngrams_into_tokens(tokens, filtered_ngrams, max_n)
    
    # Step 4: Count final n-gram lengths
    ngram_counts_by_length = {}
    for n in range(1, max_n + 1):
        if n in filtered_ngrams:
            ngram_counts_by_length[n] = len(filtered_ngrams[n])
        else:
            ngram_counts_by_length[n] = 0
    
    print(f"✅ N-gram building complete: {len(merged_tokens)} tokens")
    print(f"   Detected n-gram lengths: {list(range(1, max([k for k, v in ngram_counts_by_length.items() if v > 0]) + 1))}")
    
    return merged_tokens, ngram_counts_by_length


def _extract_ngrams(tokens: List[str], n: int) -> Counter:
    """Extract n-grams from token sequence"""
    if n == 1:
        return Counter(tokens)
    
    ngrams = []
    for i in range(len(tokens) - n + 1):
        ngram = tuple(tokens[i:i+n])
        ngrams.append(ngram)
    
    return Counter(ngrams)


def _filter_by_pmi(ngram_counts: Dict[tuple, int],
                   subgram_counts: Dict[tuple, int], 
                   unigram_counts: Dict[str, int],
                   threshold: float) -> Dict[tuple, int]:
    """
    Filter n-grams by Pointwise Mutual Information
    
    PMI(x,y) = log(P(x,y) / (P(x) * P(y)))
    For n-grams: PMI = log(freq(ngram) * N / (freq(left) * freq(right)))
    """
    if not ngram_counts:
        return {}
    
    filtered = {}
    total_unigrams = sum(unigram_counts.values())
    
    for ngram, freq in ngram_counts.items():
        if len(ngram) < 2:
            filtered[ngram] = freq
            continue
        
        # Calculate PMI for bigrams and extend for longer n-grams
        if len(ngram) == 2:
            left, right = ngram
            left_freq = unigram_counts.get(left, 0)
            right_freq = unigram_counts.get(right, 0)
            
            if left_freq > 0 and right_freq > 0:
                # PMI calculation
                pmi = math.log((freq * total_unigrams) / (left_freq * right_freq))
                if pmi >= threshold:
                    filtered[ngram] = freq
        else:
            # For n > 2, use decomposition approach
            # Check if all sub-bigrams pass PMI threshold
            valid = True
            for i in range(len(ngram) - 1):
                bigram = (ngram[i], ngram[i+1])
                left_freq = unigram_counts.get(ngram[i], 0)
                right_freq = unigram_counts.get(ngram[i+1], 0)
                
                if left_freq > 0 and right_freq > 0:
                    # Estimate bigram frequency (conservative approach)
                    bigram_freq = min(freq, left_freq, right_freq)
                    pmi = math.log((bigram_freq * total_unigrams) / (left_freq * right_freq))
                    if pmi < threshold * 0.8:  # Slightly relaxed for longer n-grams
                        valid = False
                        break
                else:
                    valid = False
                    break
            
            if valid:
                filtered[ngram] = freq
    
    return filtered


def _filter_by_llr(ngram_counts: Dict[tuple, int],
                   subgram_counts: Dict[tuple, int],
                   unigram_counts: Dict[str, int], 
                   threshold: float) -> Dict[tuple, int]:
    """
    Filter n-grams by Log-Likelihood Ratio
    
    LLR = 2 * (freq(ngram) * log(freq(ngram)) + ... - expected frequencies)
    """
    if not ngram_counts:
        return {}
    
    filtered = {}
    total_unigrams = sum(unigram_counts.values())
    
    for ngram, freq in ngram_counts.items():
        if len(ngram) < 2:
            filtered[ngram] = freq
            continue
        
        # Simplified LLR for bigrams
        if len(ngram) == 2:
            left, right = ngram
            left_freq = unigram_counts.get(left, 0)
            right_freq = unigram_counts.get(right, 0)
            
            if left_freq > 0 and right_freq > 0:
                # Expected frequency under independence
                expected = (left_freq * right_freq) / total_unigrams
                
                if expected > 0:
                    # Simplified LLR calculation
                    llr = 2 * freq * math.log(freq / expected)
                    if llr >= threshold:
                        filtered[ngram] = freq
        else:
            # For longer n-grams, use a heuristic based on component strength
            # Check if the n-gram is significantly more frequent than expected
            component_freqs = [unigram_counts.get(token, 0) for token in ngram]
            min_component_freq = min(component_freqs) if component_freqs else 0
            
            if min_component_freq > 0 and freq >= min_component_freq * 0.1:
                filtered[ngram] = freq
    
    return filtered


def _merge_ngrams_into_tokens(original_tokens: List[str], 
                             filtered_ngrams: Dict[int, Dict[tuple, int]], 
                             max_n: int) -> List[str]:
    """
    Merge validated n-grams back into token sequence
    Replace multi-token sequences with underscore-connected phrases
    """
    if not original_tokens:
        return []
    
    # Create a set of all valid n-grams (tuples) for efficient lookup
    valid_ngrams = set()
    for n in range(2, max_n + 1):  # Start from bigrams
        if n in filtered_ngrams:
            valid_ngrams.update(filtered_ngrams[n].keys())
    
    merged_tokens = []
    i = 0
    
    while i < len(original_tokens):
        # Try to find the longest matching n-gram starting at position i
        best_match = None
        best_length = 0
        
        for n in range(min(max_n, len(original_tokens) - i), 1, -1):  # Try longest first
            if i + n <= len(original_tokens):
                candidate = tuple(original_tokens[i:i+n])
                if candidate in valid_ngrams:
                    best_match = candidate
                    best_length = n
                    break
        
        if best_match:
            # Merge n-gram with underscores
            merged_token = "_".join(best_match)
            merged_tokens.append(merged_token)
            i += best_length
        else:
            # Keep original token
            merged_tokens.append(original_tokens[i])
            i += 1
    
    return merged_tokens


def get_ngram_stats(merged_tokens: List[str]) -> Dict[str, int]:
    """
    Get statistics about n-gram distribution in merged tokens
    
    Args:
        merged_tokens: Tokens after n-gram merging
        
    Returns:
        Dict[str, int]: Statistics by token type
    """
    stats = {
        'single_chars': 0,      # Single Chinese characters
        'two_chars': 0,         # Two-character words
        'three_chars': 0,       # Three-character phrases
        'four_chars': 0,        # Four-character idioms
        'multi_chars': 0,       # Longer phrases
        'english_terms': 0,     # English words
        'compound_ngrams': 0,   # Merged n-grams (containing underscores)
        'total_tokens': len(merged_tokens)
    }
    
    for token in merged_tokens:
        if '_' in token:
            stats['compound_ngrams'] += 1
        elif _is_english_token(token):
            stats['english_terms'] += 1
        elif _is_chinese_token(token):
            char_count = len(token)
            if char_count == 1:
                stats['single_chars'] += 1
            elif char_count == 2:
                stats['two_chars'] += 1
            elif char_count == 3:
                stats['three_chars'] += 1
            elif char_count == 4:
                stats['four_chars'] += 1
            else:
                stats['multi_chars'] += 1
    
    return stats


def _is_chinese_token(token: str) -> bool:
    """Check if token is primarily Chinese"""
    if not token:
        return False
    chinese_chars = sum(1 for char in token if '\u4e00' <= char <= '\u9fff')
    return chinese_chars > len(token) * 0.5


def _is_english_token(token: str) -> bool:
    """Check if token is English"""
    if not token:
        return False
    return bool(re.match(r'^[a-zA-Z][\w\-_]*$', token))


import re  # Add this import at the top if not already present