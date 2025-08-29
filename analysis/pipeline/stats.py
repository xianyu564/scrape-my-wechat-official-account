#!/usr/bin/env python3
"""
Statistical analysis: frequency, TF-IDF, Zipf's law, Heaps' law, lexical metrics
"""

import math
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.feature_extraction.text import TfidfVectorizer
from typing import List, Dict, Tuple, Counter as CounterType, Any, Optional
from collections import Counter
import warnings


def calculate_frequencies(corpus_tokens: List[List[str]]) -> Counter:
    """
    Calculate overall term frequencies
    
    Args:
        corpus_tokens: List of tokenized documents
        
    Returns:
        Counter: Overall term frequencies
    """
    overall_counter = Counter()
    for tokens in corpus_tokens:
        overall_counter.update(tokens)
    
    return overall_counter


def calculate_frequencies_by_year(corpus_by_year: Dict[str, List[List[str]]]) -> Dict[str, Counter]:
    """
    Calculate term frequencies by year
    
    Args:
        corpus_by_year: Tokenized documents grouped by year
        
    Returns:
        Dict[str, Counter]: Frequencies by year
    """
    freq_by_year = {}
    
    for year, year_tokens in corpus_by_year.items():
        year_counter = Counter()
        for tokens in year_tokens:
            year_counter.update(tokens)
        freq_by_year[year] = year_counter
    
    return freq_by_year


def calculate_tfidf(texts_by_year: Dict[str, List[str]], 
                   tokenizer_func, 
                   min_df: int = 1, 
                   max_df: float = 0.98,
                   max_features: Optional[int] = None,
                   topk: int = 100) -> pd.DataFrame:
    """
    Calculate TF-IDF scores using scikit-learn with pre-tokenized input
    
    Args:
        texts_by_year: Raw texts grouped by year
        tokenizer_func: Tokenization function
        min_df: Minimum document frequency
        max_df: Maximum document frequency  
        max_features: Maximum number of features
        topk: Top K terms per year
        
    Returns:
        pd.DataFrame: TF-IDF results with columns [year, word, score]
    """
    if not texts_by_year:
        return pd.DataFrame(columns=['year', 'word', 'score'])
    
    print(f"🔍 Calculating TF-IDF: min_df={min_df}, max_df={max_df}")
    
    results = []
    
    try:
        # Create TF-IDF vectorizer
        vectorizer = TfidfVectorizer(
            tokenizer=tokenizer_func,
            lowercase=False,  # We handle this in tokenizer
            min_df=min_df,
            max_df=max_df,
            max_features=max_features,
            stop_words=None,  # We handle this in tokenizer
            token_pattern=None  # Use custom tokenizer
        )
        
        for year, texts in texts_by_year.items():
            if not texts:
                continue
                
            try:
                # Fit and transform texts for this year
                tfidf_matrix = vectorizer.fit_transform(texts)
                feature_names = vectorizer.get_feature_names_out()
                
                # Calculate mean TF-IDF scores for each term
                mean_scores = np.mean(tfidf_matrix.toarray(), axis=0)
                
                # Get top K terms
                top_indices = np.argsort(mean_scores)[-topk:][::-1]
                
                for idx in top_indices:
                    if mean_scores[idx] > 0:  # Only include terms with positive scores
                        results.append({
                            'year': year,
                            'word': feature_names[idx],
                            'score': mean_scores[idx]
                        })
                        
            except Exception as e:
                warnings.warn(f"TF-IDF calculation failed for year {year}: {e}")
                continue
    
    except Exception as e:
        warnings.warn(f"TF-IDF vectorizer setup failed: {e}")
        return pd.DataFrame(columns=['year', 'word', 'score'])
    
    print(f"✅ TF-IDF complete: {len(results)} term-year pairs")
    return pd.DataFrame(results)


def analyze_zipf_law(frequencies: Counter) -> Dict[str, float]:
    """
    Analyze Zipf's law: rank-frequency relationship
    
    Args:
        frequencies: Term frequency counter
        
    Returns:
        Dict: Zipf analysis results (slope, r_squared, etc.)
    """
    if not frequencies:
        return {'slope': 0, 'r_squared': 0, 'intercept': 0}
    
    # Sort by frequency (descending)
    sorted_freqs = sorted(frequencies.values(), reverse=True)
    
    if len(sorted_freqs) < 10:  # Need minimum data points
        return {'slope': 0, 'r_squared': 0, 'intercept': 0}
    
    # Create rank and frequency arrays (log scale)
    ranks = np.arange(1, len(sorted_freqs) + 1)
    freqs = np.array(sorted_freqs)
    
    # Filter out zero frequencies
    valid_indices = freqs > 0
    ranks = ranks[valid_indices]
    freqs = freqs[valid_indices]
    
    if len(ranks) < 10:
        return {'slope': 0, 'r_squared': 0, 'intercept': 0}
    
    # Log-log regression: log(freq) = a * log(rank) + b
    log_ranks = np.log(ranks)
    log_freqs = np.log(freqs)
    
    try:
        slope, intercept, r_value, p_value, std_err = stats.linregress(log_ranks, log_freqs)
        
        return {
            'slope': slope,
            'intercept': intercept,
            'r_squared': r_value ** 2,
            'p_value': p_value,
            'std_err': std_err,
            'total_terms': len(frequencies),
            'valid_terms': len(ranks)
        }
    except Exception as e:
        warnings.warn(f"Zipf analysis failed: {e}")
        return {'slope': 0, 'r_squared': 0, 'intercept': 0}


def analyze_heaps_law(corpus_tokens: List[List[str]]) -> Dict[str, float]:
    """
    Analyze Heaps' law: vocabulary growth with corpus size
    V = K * n^β where V=vocabulary size, n=corpus size
    
    Args:
        corpus_tokens: List of tokenized documents
        
    Returns:
        Dict: Heaps analysis results (K, beta, r_squared)
    """
    if not corpus_tokens:
        return {'K': 0, 'beta': 0, 'r_squared': 0}
    
    # Calculate cumulative vocabulary size
    vocabulary = set()
    corpus_sizes = []
    vocab_sizes = []
    
    total_tokens = 0
    for tokens in corpus_tokens:
        total_tokens += len(tokens)
        vocabulary.update(tokens)
        
        corpus_sizes.append(total_tokens)
        vocab_sizes.append(len(vocabulary))
    
    if len(corpus_sizes) < 10:
        return {'K': 0, 'beta': 0, 'r_squared': 0}
    
    # Convert to numpy arrays
    n = np.array(corpus_sizes)
    V = np.array(vocab_sizes)
    
    # Filter out zeros and small values
    valid_indices = (n > 0) & (V > 0)
    n = n[valid_indices]
    V = V[valid_indices]
    
    if len(n) < 10:
        return {'K': 0, 'beta': 0, 'r_squared': 0}
    
    # Log-log regression: log(V) = log(K) + β * log(n)
    log_n = np.log(n)
    log_V = np.log(V)
    
    try:
        beta, log_K, r_value, p_value, std_err = stats.linregress(log_n, log_V)
        K = math.exp(log_K)
        
        return {
            'K': K,
            'beta': beta,
            'r_squared': r_value ** 2,
            'p_value': p_value,
            'std_err': std_err,
            'total_documents': len(corpus_tokens),
            'final_corpus_size': corpus_sizes[-1],
            'final_vocab_size': vocab_sizes[-1]
        }
    except Exception as e:
        warnings.warn(f"Heaps analysis failed: {e}")
        return {'K': 0, 'beta': 0, 'r_squared': 0}


def calculate_lexical_metrics(corpus_tokens: List[List[str]]) -> Dict[str, float]:
    """
    Calculate lexical diversity and density metrics
    
    Args:
        corpus_tokens: List of tokenized documents
        
    Returns:
        Dict: Lexical metrics (TTR, lexical_density, etc.)
    """
    if not corpus_tokens:
        return {}
    
    # Flatten all tokens
    all_tokens = []
    for tokens in corpus_tokens:
        all_tokens.extend(tokens)
    
    if not all_tokens:
        return {}
    
    # Basic counts
    total_tokens = len(all_tokens)
    unique_tokens = len(set(all_tokens))
    
    # Type-Token Ratio (TTR)
    ttr = unique_tokens / total_tokens if total_tokens > 0 else 0
    
    # Lexical density (content words ratio)
    # Simplified: assume tokens without underscores or length > 1 are content words
    content_tokens = [token for token in all_tokens 
                     if len(token) > 1 or '_' in token or _is_content_word(token)]
    lexical_density = len(content_tokens) / total_tokens if total_tokens > 0 else 0
    
    # Average document length
    doc_lengths = [len(tokens) for tokens in corpus_tokens]
    avg_doc_length = np.mean(doc_lengths) if doc_lengths else 0
    
    return {
        'total_tokens': total_tokens,
        'unique_tokens': unique_tokens,
        'ttr': ttr,
        'lexical_density': lexical_density,
        'avg_doc_length': avg_doc_length,
        'total_documents': len(corpus_tokens)
    }


def get_year_over_year_growth(freq_by_year: Dict[str, Counter], topk: int = 20) -> List[Dict[str, Any]]:
    """
    Calculate year-over-year word frequency growth
    
    Args:
        freq_by_year: Frequencies by year
        topk: Top K growing terms
        
    Returns:
        List[Dict]: YoY growth data
    """
    if len(freq_by_year) < 2:
        return []
    
    years = sorted(freq_by_year.keys())
    growth_data = []
    
    for i in range(1, len(years)):
        prev_year = years[i-1]
        curr_year = years[i]
        
        prev_freq = freq_by_year[prev_year]
        curr_freq = freq_by_year[curr_year]
        
        # Calculate growth for each word
        word_growth = {}
        for word in set(prev_freq.keys()) | set(curr_freq.keys()):
            prev_count = prev_freq.get(word, 0)
            curr_count = curr_freq.get(word, 0)
            growth = curr_count - prev_count
            
            if abs(growth) > 0:  # Only include words with change
                word_growth[word] = {
                    'word': word,
                    'prev_year': prev_year,
                    'curr_year': curr_year,
                    'prev_count': prev_count,
                    'curr_count': curr_count,
                    'growth': growth
                }
        
        # Sort by growth and take top K
        sorted_growth = sorted(word_growth.values(), key=lambda x: x['growth'], reverse=True)
        growth_data.extend(sorted_growth[:topk])
    
    return growth_data


def _is_content_word(token: str) -> bool:
    """Check if token is likely a content word (simplified heuristic)"""
    if not token:
        return False
    
    # Chinese content words (heuristic: meaningful single characters)
    if len(token) == 1 and '\u4e00' <= token <= '\u9fff':
        # Common meaningful single Chinese characters
        content_chars = {'人', '心', '光', '火', '爱', '情', '美', '善', '真', '智', '慧'}
        return token in content_chars
    
    # Multi-character words are likely content words
    return len(token) > 1


def save_summary_stats(freq_overall: Counter,
                      freq_by_year: Dict[str, Counter], 
                      tfidf_results: pd.DataFrame,
                      zipf_results: Dict[str, float],
                      heaps_results: Dict[str, float],
                      lexical_metrics: Dict[str, float],
                      ngram_stats: Dict[str, int],
                      output_path: str) -> None:
    """
    Save comprehensive summary statistics to JSON
    
    Args:
        freq_overall: Overall frequencies
        freq_by_year: Frequencies by year
        tfidf_results: TF-IDF results DataFrame
        zipf_results: Zipf law analysis results
        heaps_results: Heaps law analysis results  
        lexical_metrics: Lexical diversity metrics
        ngram_stats: N-gram distribution statistics
        output_path: Output JSON file path
    """
    import json
    
    summary = {
        'overall_stats': {
            'total_unique_words': len(freq_overall),
            'total_word_frequency': sum(freq_overall.values()),
            'top_20_words': [{'word': word, 'freq': freq} 
                           for word, freq in freq_overall.most_common(20)]
        },
        'yearly_stats': {
            'years': sorted(freq_by_year.keys()),
            'words_per_year': {year: len(counter) for year, counter in freq_by_year.items()},
            'total_freq_per_year': {year: sum(counter.values()) for year, counter in freq_by_year.items()}
        },
        'tfidf_stats': {
            'total_term_year_pairs': len(tfidf_results),
            'years_with_tfidf': sorted(tfidf_results['year'].unique().tolist()) if not tfidf_results.empty else []
        },
        'zipf_analysis': zipf_results,
        'heaps_analysis': heaps_results,
        'lexical_metrics': lexical_metrics,
        'ngram_stats': ngram_stats,
        'ngram_lengths_detected': [n for n, count in ngram_stats.items() 
                                 if isinstance(n, int) and count > 0] if ngram_stats else []
    }
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    
    print(f"📊 Summary statistics saved to: {output_path}")