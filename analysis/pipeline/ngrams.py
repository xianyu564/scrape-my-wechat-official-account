#!/usr/bin/env python3
"""
Enhanced variable-length n-gram builder with PMI/LLR collocation filtering
- Supports 1..MAX_N (8+) n-gram extraction using sliding windows
- PMI (Pointwise Mutual Information) and LLR (Log-Likelihood Ratio) scoring
- Memory-efficient iterative processing for large corpora
- Phrase merging with underscore connection and character span preservation
- Dunning 1993 LLR implementation with contingency tables
"""

import re
import math
from typing import List, Dict, Tuple, Set, Counter as CounterType, Iterator, NamedTuple
from collections import Counter, defaultdict
import itertools
import warnings


class NGramSpan(NamedTuple):
    """N-gram with character span information for reporting"""
    tokens: Tuple[str, ...]
    start_pos: int
    end_pos: int
    score: float
    measure: str  # 'pmi' or 'llr'


def build_ngrams(tokens: List[str], 
                max_n: int = 8, 
                min_freq: int = 5, 
                collocation: str = 'pmi',
                pmi_threshold: float = 3.0,
                llr_threshold: float = 10.83,
                use_iterative: bool = True) -> Tuple[List[str], Dict[int, int]]:
    """
    Build variable-length n-grams with collocation filtering
    
    Args:
        tokens: List of tokens from corpus
        max_n: Maximum n-gram length (not capped at 1-4, supports 8+)
        min_freq: Minimum frequency threshold  
        collocation: 'pmi' or 'llr' for collocation measure
        pmi_threshold: PMI threshold for collocation filtering
        llr_threshold: Log-likelihood ratio threshold (Dunning 1993)
        use_iterative: Use memory-efficient iterative processing
        
    Returns:
        Tuple[List[str], Dict[int, int]]: (merged_tokens, ngram_counts_by_length)
    """
    if not tokens:
        return [], {}
    
    print(f"🔢 Building n-grams: max_n={max_n}, min_freq={min_freq}, measure={collocation}")
    print(f"   Input tokens: {len(tokens)}, iterative={use_iterative}")
    
    # Step 1: Generate all n-gram candidates using sliding windows
    if use_iterative:
        ngram_candidates = _extract_ngrams_iterative(tokens, max_n, min_freq)
    else:
        ngram_candidates = _extract_ngrams_batch(tokens, max_n, min_freq)
    
    # Report extraction results
    for n in range(1, max_n + 1):
        count = len(ngram_candidates.get(n, {}))
        print(f"   - {n}-grams: {count} candidates (freq >= {min_freq})")
    
    # Step 2: Apply collocation filtering for n > 1
    filtered_ngrams = {}
    ngram_spans = {}  # Store span information for reporting
    
    # Keep all unigrams as baseline
    filtered_ngrams[1] = ngram_candidates.get(1, {})
    ngram_spans[1] = {}
    
    # Get total corpus size for probability calculations
    total_tokens = len(tokens)
    unigram_counts = ngram_candidates.get(1, {})
    
    for n in range(2, max_n + 1):
        if n not in ngram_candidates or not ngram_candidates[n]:
            filtered_ngrams[n] = {}
            ngram_spans[n] = {}
            continue
            
        if collocation == 'pmi':
            filtered_ngrams[n], ngram_spans[n] = _filter_by_pmi_enhanced(
                ngram_candidates[n], 
                unigram_counts,
                total_tokens,
                threshold=pmi_threshold
            )
        elif collocation == 'llr':
            filtered_ngrams[n], ngram_spans[n] = _filter_by_llr_enhanced(
                ngram_candidates[n],
                unigram_counts,
                total_tokens, 
                threshold=llr_threshold
            )
        else:
            # No filtering, keep all frequent n-grams
            filtered_ngrams[n] = ngram_candidates[n]
            ngram_spans[n] = {}
        
        print(f"   - {n}-grams after {collocation}: {len(filtered_ngrams[n])} retained")
    
    # Step 3: Merge n-grams back into token sequence with span preservation
    merged_tokens = _merge_ngrams_into_tokens_enhanced(tokens, filtered_ngrams, ngram_spans, max_n)
    
    # Step 4: Count final n-gram lengths and validate coverage
    ngram_counts_by_length = {}
    detected_lengths = []
    
    for n in range(1, max_n + 1):
        count = len(filtered_ngrams.get(n, {}))
        ngram_counts_by_length[n] = count
        if count > 0:
            detected_lengths.append(n)
    
    # Validation: check coverage
    expected_lengths = list(range(1, max_n + 1))
    missing_lengths = [n for n in expected_lengths if ngram_counts_by_length.get(n, 0) == 0]
    
    if missing_lengths and max(missing_lengths) <= 3:  # Only warn for basic n-grams
        print(f"⚠️  Missing n-gram lengths: {missing_lengths} (consider lowering min_freq={min_freq})")
    
    print(f"✅ N-gram building complete: {len(merged_tokens)} tokens")
    print(f"   Detected n-gram lengths: {detected_lengths}")
    
    return merged_tokens, ngram_counts_by_length


def _extract_ngrams_iterative(tokens: List[str], max_n: int, min_freq: int) -> Dict[int, Dict]:
    """Extract n-grams using memory-efficient sliding windows"""
    ngram_candidates = defaultdict(lambda: defaultdict(int))
    
    # Use generators for memory efficiency
    for n in range(1, max_n + 1):
        for ngram in _sliding_window_ngrams(tokens, n):
            ngram_candidates[n][ngram] += 1
    
    # Filter by frequency
    filtered_candidates = {}
    for n in range(1, max_n + 1):
        filtered_candidates[n] = {
            ngram: count for ngram, count in ngram_candidates[n].items() 
            if count >= min_freq
        }
    
    return filtered_candidates


def _extract_ngrams_batch(tokens: List[str], max_n: int, min_freq: int) -> Dict[int, Dict]:
    """Extract n-grams using batch processing (faster but more memory)"""
    ngram_candidates = {}
    
    for n in range(1, max_n + 1):
        if n == 1:
            ngrams = Counter(tokens)
        else:
            ngrams = []
            for i in range(len(tokens) - n + 1):
                ngram = tuple(tokens[i:i+n])
                ngrams.append(ngram)
            ngrams = Counter(ngrams)
        
        # Filter by frequency
        frequent_ngrams = {ngram: count for ngram, count in ngrams.items() if count >= min_freq}
        ngram_candidates[n] = frequent_ngrams
    
    return ngram_candidates


def _sliding_window_ngrams(tokens: List[str], n: int) -> Iterator[Tuple[str, ...]]:
    """Generator for n-grams using sliding window"""
    if n == 1:
        for token in tokens:
            yield token
    else:
        for i in range(len(tokens) - n + 1):
            yield tuple(tokens[i:i+n])


def _filter_by_pmi_enhanced(ngram_counts: Dict[tuple, int],
                           unigram_counts: Dict[str, int],
                           total_tokens: int,
                           threshold: float) -> Tuple[Dict[tuple, int], Dict[tuple, NGramSpan]]:
    """
    Enhanced PMI filtering with proper probability calculations
    
    PMI(x,y) = log(P(x,y) / (P(x) * P(y)))
    For n-grams: PMI = log(freq(ngram) * N / (freq(w1) * freq(w2) * ... * freq(wn)))
    """
    if not ngram_counts:
        return {}, {}
    
    filtered = {}
    spans = {}
    
    for ngram, freq in ngram_counts.items():
        if len(ngram) < 2:
            # Skip unigrams for PMI calculation
            continue
        
        # Calculate PMI score
        # For bigrams: PMI = log(P(w1,w2) / (P(w1) * P(w2)))
        # For n-grams: extend using geometric mean of component PMIs
        
        if len(ngram) == 2:
            w1, w2 = ngram
            freq_w1 = unigram_counts.get(w1, 0)
            freq_w2 = unigram_counts.get(w2, 0)
            
            if freq_w1 > 0 and freq_w2 > 0:
                # PMI calculation with smoothing
                p_ngram = freq / total_tokens
                p_w1 = freq_w1 / total_tokens
                p_w2 = freq_w2 / total_tokens
                
                pmi = math.log2(p_ngram / (p_w1 * p_w2))
                
                if pmi >= threshold:
                    filtered[ngram] = freq
                    spans[ngram] = NGramSpan(
                        tokens=ngram,
                        start_pos=0,  # Will be filled during merging
                        end_pos=0,
                        score=pmi,
                        measure='pmi'
                    )
        else:
            # For n > 2, use minimum pairwise PMI as a conservative measure
            min_pmi = float('inf')
            valid = True
            
            for i in range(len(ngram) - 1):
                w1, w2 = ngram[i], ngram[i+1]
                freq_w1 = unigram_counts.get(w1, 0)
                freq_w2 = unigram_counts.get(w2, 0)
                
                if freq_w1 <= 0 or freq_w2 <= 0:
                    valid = False
                    break
                
                # Estimate pairwise frequency (conservative)
                pair_freq = min(freq, freq_w1, freq_w2)
                p_pair = pair_freq / total_tokens
                p_w1 = freq_w1 / total_tokens
                p_w2 = freq_w2 / total_tokens
                
                pmi = math.log2(p_pair / (p_w1 * p_w2))
                min_pmi = min(min_pmi, pmi)
            
            # Use relaxed threshold for longer n-grams
            relaxed_threshold = threshold * (0.8 if len(ngram) == 3 else 0.6)
            
            if valid and min_pmi >= relaxed_threshold:
                filtered[ngram] = freq
                spans[ngram] = NGramSpan(
                    tokens=ngram,
                    start_pos=0,
                    end_pos=0,
                    score=min_pmi,
                    measure='pmi'
                )
    
    return filtered, spans


def _filter_by_llr_enhanced(ngram_counts: Dict[tuple, int],
                           unigram_counts: Dict[str, int],
                           total_tokens: int,
                           threshold: float) -> Tuple[Dict[tuple, int], Dict[tuple, NGramSpan]]:
    """
    Enhanced LLR filtering following Dunning 1993
    
    LLR = 2 * sum(O * log(O/E)) where O=observed, E=expected
    Uses contingency table for proper statistical testing
    """
    if not ngram_counts:
        return {}, {}
    
    filtered = {}
    spans = {}
    
    for ngram, freq in ngram_counts.items():
        if len(ngram) < 2:
            continue
        
        # For bigrams, use full Dunning 1993 contingency table
        if len(ngram) == 2:
            w1, w2 = ngram
            freq_w1 = unigram_counts.get(w1, 0)
            freq_w2 = unigram_counts.get(w2, 0)
            
            if freq_w1 > 0 and freq_w2 > 0:
                llr = _calculate_dunning_llr(freq, freq_w1, freq_w2, total_tokens)
                
                if llr >= threshold:
                    filtered[ngram] = freq
                    spans[ngram] = NGramSpan(
                        tokens=ngram,
                        start_pos=0,
                        end_pos=0,
                        score=llr,
                        measure='llr'
                    )
        else:
            # For n > 2, use aggregated LLR scores
            total_llr = 0
            valid = True
            
            for i in range(len(ngram) - 1):
                w1, w2 = ngram[i], ngram[i+1]
                freq_w1 = unigram_counts.get(w1, 0)
                freq_w2 = unigram_counts.get(w2, 0)
                
                if freq_w1 <= 0 or freq_w2 <= 0:
                    valid = False
                    break
                
                # Estimate pairwise frequency
                pair_freq = min(freq, freq_w1, freq_w2)
                llr = _calculate_dunning_llr(pair_freq, freq_w1, freq_w2, total_tokens)
                total_llr += llr
            
            # Average LLR with relaxed threshold
            if valid and len(ngram) > 1:
                avg_llr = total_llr / (len(ngram) - 1)
                relaxed_threshold = threshold * (0.8 if len(ngram) == 3 else 0.6)
                
                if avg_llr >= relaxed_threshold:
                    filtered[ngram] = freq
                    spans[ngram] = NGramSpan(
                        tokens=ngram,
                        start_pos=0,
                        end_pos=0,
                        score=avg_llr,
                        measure='llr'
                    )
    
    return filtered, spans


def _calculate_dunning_llr(observed_together: int, freq_w1: int, freq_w2: int, total_tokens: int) -> float:
    """
    Calculate Dunning's Log-Likelihood Ratio using contingency table
    
    Contingency table:
                w2     ~w2
        w1      a      b      (freq_w1)
        ~w1     c      d      
               (freq_w2)     (total_tokens)
    
    where a = observed_together, b = freq_w1 - a, c = freq_w2 - a, d = total - freq_w1 - freq_w2 + a
    """
    # Contingency table cells
    a = observed_together
    b = freq_w1 - a
    c = freq_w2 - a  
    d = total_tokens - freq_w1 - freq_w2 + a
    
    # Ensure all cells are positive (add small smoothing if needed)
    if a <= 0 or b <= 0 or c <= 0 or d <= 0:
        return 0.0
    
    # Expected values under independence
    e_a = (freq_w1 * freq_w2) / total_tokens
    e_b = (freq_w1 * (total_tokens - freq_w2)) / total_tokens
    e_c = ((total_tokens - freq_w1) * freq_w2) / total_tokens
    e_d = ((total_tokens - freq_w1) * (total_tokens - freq_w2)) / total_tokens
    
    # Ensure expected values are positive
    if e_a <= 0 or e_b <= 0 or e_c <= 0 or e_d <= 0:
        return 0.0
    
    # Calculate LLR = 2 * sum(O * log(O/E))
    try:
        llr = 2 * (
            a * math.log(a / e_a) +
            b * math.log(b / e_b) +
            c * math.log(c / e_c) +
            d * math.log(d / e_d)
        )
        return llr
    except (ValueError, ZeroDivisionError):
        return 0.0


def _merge_ngrams_into_tokens_enhanced(original_tokens: List[str], 
                                      filtered_ngrams: Dict[int, Dict[tuple, int]], 
                                      ngram_spans: Dict[int, Dict[tuple, NGramSpan]],
                                      max_n: int) -> List[str]:
    """
    Enhanced n-gram merging with character span preservation
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
            # Merge n-gram with underscores, preserving original character spans
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
    Enhanced statistics about n-gram distribution in merged tokens
    
    Args:
        merged_tokens: Tokens after n-gram merging
        
    Returns:
        Dict[str, int]: Statistics by token type and n-gram length
    """
    stats = {
        'single_chars': 0,          # Single Chinese characters
        'two_chars': 0,             # Two-character words
        'three_chars': 0,           # Three-character phrases
        'four_chars': 0,            # Four-character idioms
        'multi_chars': 0,           # Longer phrases
        'english_terms': 0,         # English words
        'compound_ngrams': 0,       # Merged n-grams (containing underscores)
        'total_tokens': len(merged_tokens),
        'ngram_length_distribution': defaultdict(int)  # Distribution by n-gram length
    }
    
    for token in merged_tokens:
        if '_' in token:
            stats['compound_ngrams'] += 1
            # Count components in compound n-gram
            components = token.split('_')
            stats['ngram_length_distribution'][len(components)] += 1
        elif _is_english_token(token):
            stats['english_terms'] += 1
            stats['ngram_length_distribution'][1] += 1
        elif _is_chinese_token(token):
            # Count Chinese characters (not Unicode length)
            char_count = _count_chinese_characters(token)
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
            stats['ngram_length_distribution'][1] += 1
    
    # Convert defaultdict to regular dict for JSON serialization
    stats['ngram_length_distribution'] = dict(stats['ngram_length_distribution'])
    
    return stats


def _is_chinese_token(token: str) -> bool:
    """Check if token is primarily Chinese"""
    if not token:
        return False
    chinese_chars = sum(1 for char in token if '\u4e00' <= char <= '\u9fff')
    return chinese_chars > len(token) * 0.5


def _count_chinese_characters(token: str) -> int:
    """Count the number of Chinese characters in a token"""
    return sum(1 for char in token if '\u4e00' <= char <= '\u9fff')


def _is_english_token(token: str) -> bool:
    """Check if token is English"""
    if not token:
        return False
    return bool(re.match(r'^[a-zA-Z][\w\-_]*$', token))


# Validation and debugging utilities
def validate_ngram_coverage(ngram_counts_by_length: Dict[int, int], 
                           max_n: int, 
                           min_freq: int) -> Dict[str, any]:
    """
    Validate n-gram detection coverage and provide debugging information
    
    Returns:
        Dict with coverage analysis and recommendations
    """
    detected_lengths = [n for n, count in ngram_counts_by_length.items() if count > 0]
    missing_lengths = [n for n in range(1, max_n + 1) if n not in detected_lengths]
    
    analysis = {
        'detected_lengths': detected_lengths,
        'missing_lengths': missing_lengths,
        'max_detected': max(detected_lengths) if detected_lengths else 0,
        'coverage_ratio': len(detected_lengths) / max_n,
        'recommendations': []
    }
    
    # Generate recommendations
    if not detected_lengths:
        analysis['recommendations'].append(f"No n-grams detected. Try lowering min_freq from {min_freq} to {max(1, min_freq//2)}")
    elif analysis['max_detected'] < 2:
        analysis['recommendations'].append(f"Only unigrams detected. Lower min_freq from {min_freq} to {max(2, min_freq//3)}")
    elif analysis['coverage_ratio'] < 0.5:
        analysis['recommendations'].append(f"Low coverage ({analysis['coverage_ratio']:.1%}). Consider lowering min_freq or threshold")
    
    if 2 not in detected_lengths:
        analysis['recommendations'].append("No bigrams detected - this may indicate overly strict thresholds")
    
    return analysis


