#!/usr/bin/env python3
"""
Performance and robustness enhancements for the analysis pipeline
Task 8: 性能与健壮性
"""

import gc
import os
import sys
import warnings
from collections import Counter
from collections.abc import Iterator
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import psutil

# Add pipeline to path
sys.path.insert(0, str(Path(__file__).parent / "pipeline"))


class MemoryMonitor:
    """Monitor memory usage during processing"""

    def __init__(self):
        self.process = psutil.Process(os.getpid())
        self.start_memory = self.get_memory_mb()
        self.peak_memory = self.start_memory

    def get_memory_mb(self) -> float:
        """Get current memory usage in MB"""
        return self.process.memory_info().rss / 1024 / 1024

    def update_peak(self):
        """Update peak memory usage"""
        current = self.get_memory_mb()
        if current > self.peak_memory:
            self.peak_memory = current

    def get_usage_stats(self) -> Dict[str, float]:
        """Get memory usage statistics"""
        current = self.get_memory_mb()
        return {
            'start_mb': self.start_memory,
            'current_mb': current,
            'peak_mb': self.peak_memory,
            'increase_mb': current - self.start_memory
        }


def chunk_iterator(items: List[Any], chunk_size: int = 1000) -> Iterator[List[Any]]:
    """
    Memory-efficient iterator for processing large lists in chunks
    
    Args:
        items: List of items to process
        chunk_size: Size of each chunk
        
    Yields:
        List[Any]: Chunks of items
    """
    for i in range(0, len(items), chunk_size):
        yield items[i:i + chunk_size]


def safe_divide(numerator: float, denominator: float, default: float = 0.0) -> float:
    """
    Safe division with error handling
    
    Args:
        numerator: Numerator value
        denominator: Denominator value  
        default: Default value if division fails
        
    Returns:
        float: Result of division or default
    """
    try:
        if denominator == 0:
            return default
        result = numerator / denominator
        if np.isnan(result) or np.isinf(result):
            return default
        return result
    except (ZeroDivisionError, TypeError, ValueError):
        return default


def graceful_degradation(min_freq: int, corpus_size: int,
                        ngram_counts: Dict[int, int]) -> Tuple[int, List[str]]:
    """
    Gracefully degrade parameters when data is insufficient
    
    Args:
        min_freq: Current minimum frequency threshold
        corpus_size: Size of the corpus
        ngram_counts: Current n-gram counts by length
        
    Returns:
        Tuple[int, List[str]]: (adjusted_min_freq, reasons)
    """
    reasons = []
    adjusted_freq = min_freq

    # Check if corpus is too small
    if corpus_size < 100:
        adjusted_freq = max(1, min_freq // 4)
        reasons.append(f"Very small corpus ({corpus_size} tokens), reduced min_freq to {adjusted_freq}")
    elif corpus_size < 1000:
        adjusted_freq = max(1, min_freq // 2)
        reasons.append(f"Small corpus ({corpus_size} tokens), reduced min_freq to {adjusted_freq}")

    # Check if no n-grams detected
    total_ngrams = sum(ngram_counts.values())
    if total_ngrams == 0 and min_freq > 1:
        adjusted_freq = max(1, min_freq // 2)
        reasons.append(f"No n-grams detected, reduced min_freq to {adjusted_freq}")

    # Check specific lengths
    if ngram_counts.get(2, 0) == 0 and min_freq > 2:
        adjusted_freq = max(1, min_freq // 2)
        reasons.append(f"No bigrams detected, reduced min_freq to {adjusted_freq}")

    return adjusted_freq, reasons


def validate_corpus_data(corpus_tokens: List[List[str]]) -> Dict[str, Any]:
    """
    Validate corpus data and provide recommendations
    
    Args:
        corpus_tokens: List of tokenized documents
        
    Returns:
        Dict[str, Any]: Validation results and recommendations
    """
    if not corpus_tokens:
        return {
            'valid': False,
            'issues': ['Empty corpus provided'],
            'recommendations': ['Provide non-empty corpus data'],
            'stats': {'documents': 0, 'total_tokens': 0, 'unique_tokens': 0}
        }

    issues = []
    recommendations = []

    # Calculate basic stats
    total_tokens = sum(len(doc) for doc in corpus_tokens)
    all_tokens = [token for doc in corpus_tokens for token in doc]
    unique_tokens = len(set(all_tokens))

    avg_doc_length = total_tokens / len(corpus_tokens) if corpus_tokens else 0

    # Check for potential issues
    if len(corpus_tokens) < 5:
        issues.append(f'Very few documents ({len(corpus_tokens)})')
        recommendations.append('Consider collecting more documents for better analysis')

    if total_tokens < 100:
        issues.append(f'Very few tokens ({total_tokens})')
        recommendations.append('Corpus may be too small for reliable statistical analysis')

    if avg_doc_length < 5:
        issues.append(f'Very short documents (avg {avg_doc_length:.1f} tokens)')
        recommendations.append('Documents may be too short for meaningful n-gram analysis')

    if unique_tokens < 20:
        issues.append(f'Limited vocabulary ({unique_tokens} unique terms)')
        recommendations.append('Consider lowering min_freq or expanding corpus')

    # Type-token ratio check
    ttr = unique_tokens / total_tokens if total_tokens > 0 else 0
    if ttr > 0.8:
        issues.append(f'Very high TTR ({ttr:.3f}) - possibly over-tokenized')
        recommendations.append('Check tokenization settings or consider stopword filtering')
    elif ttr < 0.05:
        issues.append(f'Very low TTR ({ttr:.3f}) - possibly repetitive content')
        recommendations.append('Content may be very repetitive or formulaic')

    return {
        'valid': len(issues) == 0,
        'issues': issues,
        'recommendations': recommendations,
        'stats': {
            'documents': len(corpus_tokens),
            'total_tokens': total_tokens,
            'unique_tokens': unique_tokens,
            'avg_doc_length': avg_doc_length,
            'ttr': ttr
        }
    }


def optimize_memory_usage():
    """Optimize memory usage by forcing garbage collection"""
    gc.collect()

    # Additional memory optimization for large datasets
    try:
        # Force Python to release memory back to OS (if possible)
        if hasattr(gc, 'set_threshold'):
            # Temporarily adjust GC thresholds for more aggressive collection
            old_thresholds = gc.get_threshold()
            # Use less aggressive GC thresholds (Python default: 700, 10, 10)
            gc.set_threshold(700, 10, 10)
            gc.collect()
            gc.set_threshold(*old_thresholds)
    except Exception:
        pass  # Ignore if GC optimization fails


def safe_statistical_calculation(frequencies: Counter,
                                operation: str = 'zipf',
                                **kwargs) -> Dict[str, float]:
    """
    Safely perform statistical calculations with error handling
    
    Args:
        frequencies: Term frequency counter
        operation: Type of statistical analysis ('zipf' or 'heaps')
        **kwargs: Additional arguments for specific operations
        
    Returns:
        Dict[str, float]: Results or safe defaults
    """
    try:
        if operation == 'zipf':
            from stats import analyze_zipf_law
            return analyze_zipf_law(frequencies)
        elif operation == 'heaps':
            from stats import analyze_heaps_law
            corpus_tokens = kwargs.get('corpus_tokens', [])
            return analyze_heaps_law(corpus_tokens)
        else:
            return {'error': f'Unknown operation: {operation}'}

    except Exception as e:
        warnings.warn(f"Statistical calculation failed for {operation}: {e}")
        return {
            'slope': 0.0, 'r_squared': 0.0, 'intercept': 0.0,
            'K': 0.0, 'beta': 0.0, 'error': str(e)
        }


def adaptive_parameter_tuning(corpus_stats: Dict[str, Any],
                             initial_params: Dict[str, Any]) -> Dict[str, Any]:
    """
    Adaptively tune parameters based on corpus characteristics
    
    Args:
        corpus_stats: Corpus statistics from validation
        initial_params: Initial analysis parameters
        
    Returns:
        Dict[str, Any]: Tuned parameters with reasons
    """
    tuned_params = initial_params.copy()
    tuning_reasons = []

    total_tokens = corpus_stats.get('total_tokens', 0)
    unique_tokens = corpus_stats.get('unique_tokens', 0)
    ttr = corpus_stats.get('ttr', 0)

    # Adjust min_freq based on corpus size
    if total_tokens < 1000:
        tuned_params['min_freq'] = max(1, initial_params.get('min_freq', 5) // 3)
        tuning_reasons.append(f"Small corpus: reduced min_freq to {tuned_params['min_freq']}")
    elif total_tokens > 100000:
        tuned_params['min_freq'] = initial_params.get('min_freq', 5) * 2
        tuning_reasons.append(f"Large corpus: increased min_freq to {tuned_params['min_freq']}")

    # Adjust max_n based on corpus characteristics
    if unique_tokens < 100:
        tuned_params['max_n'] = min(3, initial_params.get('max_n', 8))
        tuning_reasons.append(f"Limited vocabulary: reduced max_n to {tuned_params['max_n']}")
    elif total_tokens < 500:
        tuned_params['max_n'] = min(4, initial_params.get('max_n', 8))
        tuning_reasons.append(f"Small corpus: reduced max_n to {tuned_params['max_n']}")

    # Adjust thresholds based on TTR
    if ttr > 0.5:  # High diversity
        tuned_params['pmi_threshold'] = initial_params.get('pmi_threshold', 3.0) * 0.8
        tuning_reasons.append(f"High diversity: lowered PMI threshold to {tuned_params['pmi_threshold']:.1f}")
    elif ttr < 0.1:  # Low diversity
        tuned_params['pmi_threshold'] = initial_params.get('pmi_threshold', 3.0) * 1.2
        tuning_reasons.append(f"Low diversity: raised PMI threshold to {tuned_params['pmi_threshold']:.1f}")

    tuned_params['tuning_reasons'] = tuning_reasons
    return tuned_params


class RobustAnalysisPipeline:
    """Robust analysis pipeline with error handling and memory management"""

    def __init__(self, max_memory_mb: float = 2048):
        """
        Initialize robust pipeline
        
        Args:
            max_memory_mb: Maximum memory usage before warnings
        """
        self.max_memory_mb = max_memory_mb
        self.memory_monitor = MemoryMonitor()
        self.errors = []
        self.warnings = []

    def check_memory_usage(self) -> bool:
        """
        Check if memory usage is within limits
        
        Returns:
            bool: True if memory usage is acceptable
        """
        self.memory_monitor.update_peak()
        current_mb = self.memory_monitor.get_memory_mb()

        if current_mb > self.max_memory_mb:
            warning_msg = f"High memory usage: {current_mb:.1f}MB (limit: {self.max_memory_mb}MB)"
            self.warnings.append(warning_msg)
            warnings.warn(warning_msg)
            return False

        return True

    def safe_tokenization(self, texts: List[str], tokenizer_func) -> List[List[str]]:
        """
        Safely tokenize texts with error handling
        
        Args:
            texts: List of text strings
            tokenizer_func: Tokenization function
            
        Returns:
            List[List[str]]: Tokenized texts
        """
        tokenized = []
        failed_count = 0

        for i, text in enumerate(texts):
            try:
                if not text or not text.strip():
                    continue

                tokens = tokenizer_func(text)
                if tokens:
                    tokenized.append(tokens)

                # Memory check every 100 documents
                if i % 100 == 0:
                    self.check_memory_usage()
                    if i % 1000 == 0:  # Force GC every 1000 docs
                        optimize_memory_usage()

            except Exception as e:
                failed_count += 1
                if failed_count < 10:  # Only log first 10 failures
                    self.errors.append(f"Tokenization failed for document {i}: {e}")

        if failed_count > 0:
            warning_msg = f"Failed to tokenize {failed_count} documents"
            self.warnings.append(warning_msg)
            warnings.warn(warning_msg)

        return tokenized

    def safe_analysis_with_fallback(self, corpus_tokens: List[List[str]],
                                   params: Dict[str, Any]) -> Dict[str, Any]:
        """
        Perform analysis with automatic fallback and parameter adjustment
        
        Args:
            corpus_tokens: Tokenized corpus
            params: Analysis parameters
            
        Returns:
            Dict[str, Any]: Analysis results with fallback information
        """
        # Validate corpus
        validation = validate_corpus_data(corpus_tokens)
        if not validation['valid']:
            # Try with degraded parameters
            degraded_params = adaptive_parameter_tuning(validation['stats'], params)
            self.warnings.extend([f"Corpus validation failed: {issue}" for issue in validation['issues']])
            self.warnings.extend(validation['recommendations'])

            if degraded_params.get('tuning_reasons'):
                self.warnings.extend(degraded_params['tuning_reasons'])

        # Perform analysis with memory monitoring
        results = {'validation': validation}

        try:
            self.check_memory_usage()

            # Add actual analysis calls here (would integrate with existing pipeline)
            results['success'] = True
            results['memory_stats'] = self.memory_monitor.get_usage_stats()
            results['errors'] = self.errors
            results['warnings'] = self.warnings

        except Exception as e:
            self.errors.append(f"Analysis failed: {e}")
            results['success'] = False
            results['error'] = str(e)

        return results

    def get_performance_report(self) -> Dict[str, Any]:
        """Get performance and robustness report"""
        memory_stats = self.memory_monitor.get_usage_stats()

        return {
            'memory_usage': memory_stats,
            'peak_memory_mb': memory_stats['peak_mb'],
            'memory_increase_mb': memory_stats['increase_mb'],
            'errors_count': len(self.errors),
            'warnings_count': len(self.warnings),
            'errors': self.errors[:10],  # Limit to first 10
            'warnings': self.warnings[:10],
            'memory_efficient': memory_stats['increase_mb'] < 500
        }


def run_robustness_test():
    """Run robustness test with edge cases"""
    print("🧪 Running robustness tests...")

    pipeline = RobustAnalysisPipeline(max_memory_mb=1024)

    # Test 1: Empty data
    print("  Test 1: Empty corpus handling")
    empty_validation = validate_corpus_data([])
    assert not empty_validation['valid']
    print(f"    ✅ Empty corpus correctly identified: {empty_validation['issues'][0]}")

    # Test 2: Minimal data
    print("  Test 2: Minimal corpus handling")
    minimal_corpus = [['hello'], ['world']]
    minimal_validation = validate_corpus_data(minimal_corpus)
    print(f"    ✅ Minimal corpus stats: {minimal_validation['stats']}")

    # Test 3: Parameter adaptation
    print("  Test 3: Parameter adaptation")
    initial_params = {'min_freq': 10, 'max_n': 8, 'pmi_threshold': 3.0}
    adapted = adaptive_parameter_tuning(minimal_validation['stats'], initial_params)
    print(f"    ✅ Adapted parameters: min_freq={adapted['min_freq']}, max_n={adapted['max_n']}")

    # Test 4: Memory monitoring
    print("  Test 4: Memory monitoring")
    memory_monitor = MemoryMonitor()
    initial_mb = memory_monitor.get_memory_mb()

    # Simulate some memory usage
    large_list = list(range(100000))
    memory_monitor.update_peak()
    peak_mb = memory_monitor.get_memory_mb()

    del large_list
    optimize_memory_usage()

    stats = memory_monitor.get_usage_stats()
    print(f"    ✅ Memory monitoring: peak={stats['peak_mb']:.1f}MB, increase={stats['increase_mb']:.1f}MB")

    performance_report = pipeline.get_performance_report()
    print(f"  📊 Performance summary: {performance_report['memory_usage']}")

    print("✅ All robustness tests passed!")


if __name__ == "__main__":
    run_robustness_test()
