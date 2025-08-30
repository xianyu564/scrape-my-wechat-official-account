#!/usr/bin/env python3
"""
Advanced linguistic analysis utilities and enhanced metrics
"""

import json
import csv
import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Tuple
from collections import Counter, defaultdict
from scipy import stats
import math
from pathlib import Path
import warnings


def calculate_advanced_metrics(corpus_tokens: List[List[str]], 
                             freq_by_year: Optional[Dict[str, Counter]] = None) -> Dict[str, float]:
    """
    Calculate advanced linguistic metrics beyond basic TTR
    
    Args:
        corpus_tokens: List of tokenized documents
        freq_by_year: Optional frequency data by year for temporal analysis
        
    Returns:
        Dict: Advanced linguistic metrics
    """
    if not corpus_tokens:
        return {}
    
    # Flatten all tokens
    from itertools import chain
    all_tokens = list(chain.from_iterable(corpus_tokens))
    
    if not all_tokens:
        return {}
    
    # Basic counts
    total_tokens = len(all_tokens)
    vocabulary = set(all_tokens)
    vocab_size = len(vocabulary)
    
    # Frequency distribution
    freq_dist = Counter(all_tokens)
    sorted_freqs = sorted(freq_dist.values(), reverse=True)
    
    metrics = {}
    
    # 1. Lexical richness measures
    metrics['total_tokens'] = total_tokens
    metrics['vocabulary_size'] = vocab_size
    metrics['type_token_ratio'] = vocab_size / total_tokens
    
    # 2. Guiraud's R (vocabulary richness)
    metrics['guiraud_r'] = vocab_size / math.sqrt(total_tokens) if total_tokens > 0 else 0
    
    # 3. Herdan's C (constant of vocabulary growth)
    if total_tokens > 1 and vocab_size > 1:
        metrics['herdan_c'] = math.log(vocab_size) / math.log(total_tokens)
    else:
        metrics['herdan_c'] = 0
    
    # 4. Yule's K (vocabulary concentration)
    if len(sorted_freqs) > 1:
        sum_freq_squared = sum(f**2 for f in freq_dist.values())
        metrics['yule_k'] = 10000 * (sum_freq_squared - total_tokens) / (total_tokens**2)
    else:
        metrics['yule_k'] = 0
    
    # 5. Simpson's D (vocabulary diversity)
    if total_tokens > 1:
        sum_ni_ni_minus_1 = sum(ni * (ni - 1) for ni in freq_dist.values())
        metrics['simpson_d'] = sum_ni_ni_minus_1 / (total_tokens * (total_tokens - 1))
    else:
        metrics['simpson_d'] = 0
    
    # 6. Shannon entropy (information content)
    if total_tokens > 0:
        entropy = 0
        for freq in freq_dist.values():
            prob = freq / total_tokens
            if prob > 0:
                entropy -= prob * math.log2(prob)
        metrics['shannon_entropy'] = entropy
    else:
        metrics['shannon_entropy'] = 0
    
    # 7. Hapax legomena (words occurring once)
    hapax_count = sum(1 for freq in freq_dist.values() if freq == 1)
    metrics['hapax_legomena_ratio'] = hapax_count / vocab_size if vocab_size > 0 else 0
    
    # 8. Document-level metrics
    doc_lengths = [len(doc) for doc in corpus_tokens]
    metrics['avg_document_length'] = np.mean(doc_lengths)
    metrics['document_length_std'] = np.std(doc_lengths)
    
    # 9. Zipf law compliance
    if len(sorted_freqs) >= 10:
        ranks = np.arange(1, len(sorted_freqs) + 1)
        log_ranks = np.log(ranks)
        log_freqs = np.log(sorted_freqs)
        
        # Remove any invalid log values
        valid_mask = np.isfinite(log_ranks) & np.isfinite(log_freqs)
        if np.sum(valid_mask) >= 10:
            slope, intercept, r_value, _, _ = stats.linregress(
                log_ranks[valid_mask], log_freqs[valid_mask]
            )
            metrics['zipf_slope'] = slope
            metrics['zipf_r_squared'] = r_value**2
        else:
            metrics['zipf_slope'] = 0
            metrics['zipf_r_squared'] = 0
    else:
        metrics['zipf_slope'] = 0
        metrics['zipf_r_squared'] = 0
    
    # 10. Temporal metrics if year data is available
    if freq_by_year and len(freq_by_year) > 1:
        years = sorted(freq_by_year.keys())
        
        # Vocabulary growth over time
        cumulative_vocab = set()
        growth_rates = []
        
        for year in years:
            year_vocab = set(freq_by_year[year].keys())
            cumulative_vocab.update(year_vocab)
            
            if len(growth_rates) > 0:
                prev_size = len(cumulative_vocab) - len(year_vocab)
                growth_rate = len(year_vocab) / prev_size if prev_size > 0 else 0
                growth_rates.append(growth_rate)
        
        metrics['avg_vocabulary_growth_rate'] = np.mean(growth_rates) if growth_rates else 0
        
        # Vocabulary stability (overlap between consecutive years)
        overlaps = []
        for i in range(1, len(years)):
            vocab1 = set(freq_by_year[years[i-1]].keys())
            vocab2 = set(freq_by_year[years[i]].keys())
            overlap = len(vocab1 & vocab2) / len(vocab1 | vocab2) if vocab1 | vocab2 else 0
            overlaps.append(overlap)
        
        metrics['avg_vocabulary_stability'] = np.mean(overlaps) if overlaps else 0
    
    return metrics


def calculate_complexity_metrics(corpus_tokens: List[List[str]]) -> Dict[str, float]:
    """
    Calculate syntactic and lexical complexity metrics
    
    Args:
        corpus_tokens: List of tokenized documents
        
    Returns:
        Dict: Complexity metrics
    """
    if not corpus_tokens:
        return {}
    
    metrics = {}
    
    # Document-level statistics
    doc_lengths = [len(doc) for doc in corpus_tokens]
    
    if doc_lengths:
        metrics['mean_sentence_length'] = np.mean(doc_lengths)
        metrics['sentence_length_variance'] = np.var(doc_lengths)
        metrics['sentence_length_cv'] = (
            np.std(doc_lengths) / np.mean(doc_lengths) 
            if np.mean(doc_lengths) > 0 else 0
        )
        
        # Complexity based on length distribution
        metrics['max_sentence_length'] = max(doc_lengths)
        metrics['min_sentence_length'] = min(doc_lengths)
        
        # Quartiles for length analysis
        q25, q50, q75 = np.percentile(doc_lengths, [25, 50, 75])
        metrics['sentence_length_q25'] = q25
        metrics['sentence_length_median'] = q50
        metrics['sentence_length_q75'] = q75
        metrics['sentence_length_iqr'] = q75 - q25
    
    # Vocabulary complexity within documents
    doc_ttrs = []
    for doc in corpus_tokens:
        if len(doc) > 0:
            doc_ttr = len(set(doc)) / len(doc)
            doc_ttrs.append(doc_ttr)
    
    if doc_ttrs:
        metrics['mean_document_ttr'] = np.mean(doc_ttrs)
        metrics['document_ttr_variance'] = np.var(doc_ttrs)
    
    return metrics


def export_analysis_results(results: Dict[str, Any], output_dir: str, 
                          format: str = 'json') -> List[str]:
    """
    Export analysis results in various formats
    
    Args:
        results: Analysis results dictionary
        output_dir: Output directory path
        format: Export format ('json', 'csv', 'excel', 'all')
        
    Returns:
        List[str]: List of created file paths
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    created_files = []
    
    def _export_json():
        """Export as JSON"""
        json_path = output_path / "analysis_results.json"
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False, default=str)
        return str(json_path)
    
    def _export_csv():
        """Export metrics as CSV"""
        csv_path = output_path / "analysis_metrics.csv"
        
        # Flatten nested dictionaries for CSV export
        flat_data = []
        
        def flatten_dict(d, prefix=''):
            for key, value in d.items():
                if isinstance(value, dict):
                    flatten_dict(value, f"{prefix}{key}_" if prefix else f"{key}_")
                elif isinstance(value, (int, float, str)):
                    flat_data.append({
                        'metric': f"{prefix}{key}",
                        'value': value,
                        'type': type(value).__name__
                    })
        
        flatten_dict(results)
        
        if flat_data:
            df = pd.DataFrame(flat_data)
            df.to_csv(csv_path, index=False)
        
        return str(csv_path)
    
    def _export_excel():
        """Export as Excel with multiple sheets"""
        excel_path = output_path / "analysis_results.xlsx"
        
        with pd.ExcelWriter(excel_path, engine='openpyxl') as writer:
            # Main metrics sheet
            if 'metrics' in results:
                metrics_df = pd.DataFrame([results['metrics']]).T
                metrics_df.columns = ['Value']
                metrics_df.to_excel(writer, sheet_name='Metrics')
            
            # Zipf analysis sheet
            if 'zipf_analysis' in results:
                zipf_df = pd.DataFrame([results['zipf_analysis']]).T
                zipf_df.columns = ['Value']
                zipf_df.to_excel(writer, sheet_name='Zipf_Analysis')
            
            # Heaps analysis sheet
            if 'heaps_analysis' in results:
                heaps_df = pd.DataFrame([results['heaps_analysis']]).T
                heaps_df.columns = ['Value']
                heaps_df.to_excel(writer, sheet_name='Heaps_Analysis')
            
            # TF-IDF results
            if 'tfidf_results' in results and isinstance(results['tfidf_results'], pd.DataFrame):
                results['tfidf_results'].to_excel(writer, sheet_name='TF_IDF', index=False)
        
        return str(excel_path)
    
    # Export based on format
    if format == 'json' or format == 'all':
        created_files.append(_export_json())
    
    if format == 'csv' or format == 'all':
        created_files.append(_export_csv())
    
    if format == 'excel' or format == 'all':
        created_files.append(_export_excel())
    
    print(f"✅ Exported analysis results to {len(created_files)} files:")
    for file_path in created_files:
        print(f"   📄 {file_path}")
    
    return created_files


def validate_analysis_config(config: Dict[str, Any]) -> Tuple[bool, List[str]]:
    """
    Validate analysis configuration parameters
    
    Args:
        config: Configuration dictionary
        
    Returns:
        Tuple[bool, List[str]]: (is_valid, error_messages)
    """
    errors = []
    
    # Required fields
    required_fields = ['corpus_path', 'output_dir']
    for field in required_fields:
        if field not in config:
            errors.append(f"Missing required field: {field}")
    
    # Validate numeric ranges
    if 'min_df' in config:
        if not isinstance(config['min_df'], int) or config['min_df'] < 1:
            errors.append("min_df must be a positive integer")
    
    if 'max_df' in config:
        if not isinstance(config['max_df'], (int, float)) or not (0 < config['max_df'] <= 1):
            errors.append("max_df must be between 0 and 1")
    
    if 'topk' in config:
        if not isinstance(config['topk'], int) or config['topk'] < 1:
            errors.append("topk must be a positive integer")
    
    # Validate paths
    if 'corpus_path' in config:
        corpus_path = Path(config['corpus_path'])
        if not corpus_path.exists():
            errors.append(f"Corpus path does not exist: {config['corpus_path']}")
    
    # Validate color scheme
    if 'color_scheme' in config:
        from viz import COLOR_SCHEMES
        if config['color_scheme'] not in COLOR_SCHEMES:
            errors.append(f"Invalid color scheme: {config['color_scheme']}. "
                         f"Available: {list(COLOR_SCHEMES.keys())}")
    
    # Validate years format
    if 'years' in config and config['years']:
        if not isinstance(config['years'], list):
            errors.append("years must be a list")
        else:
            for year in config['years']:
                if not isinstance(year, str) or not year.isdigit() or len(year) != 4:
                    errors.append(f"Invalid year format: {year}. Must be 4-digit string")
    
    return len(errors) == 0, errors


def generate_analysis_report(results: Dict[str, Any], config: Dict[str, Any]) -> str:
    """
    Generate a comprehensive text report of analysis results
    
    Args:
        results: Analysis results dictionary
        config: Configuration used for analysis
        
    Returns:
        str: Formatted text report
    """
    report_lines = []
    
    report_lines.append("=" * 80)
    report_lines.append("📊 COMPREHENSIVE LINGUISTIC ANALYSIS REPORT")
    report_lines.append("=" * 80)
    
    # Configuration section
    report_lines.append("\n🔧 ANALYSIS CONFIGURATION")
    report_lines.append("-" * 40)
    for key, value in config.items():
        if isinstance(value, (str, int, float, bool)):
            report_lines.append(f"{key:.<30} {value}")
    
    # Basic statistics
    if 'corpus_stats' in results:
        stats = results['corpus_stats']
        report_lines.append("\n📈 CORPUS STATISTICS")
        report_lines.append("-" * 40)
        for key, value in stats.items():
            if isinstance(value, (int, float)):
                if isinstance(value, float):
                    report_lines.append(f"{key:.<30} {value:.3f}")
                else:
                    report_lines.append(f"{key:.<30} {value:,}")
            else:
                report_lines.append(f"{key:.<30} {value}")
    
    # Zipf law analysis
    if 'zipf_analysis' in results:
        zipf = results['zipf_analysis']
        report_lines.append("\n📊 ZIPF'S LAW ANALYSIS")
        report_lines.append("-" * 40)
        report_lines.append(f"Slope (α):........................ {zipf.get('slope', 0):.3f}")
        report_lines.append(f"R-squared:........................ {zipf.get('r_squared', 0):.3f}")
        report_lines.append(f"Intercept:........................ {zipf.get('intercept', 0):.3f}")
        
        # Interpretation
        r_squared = zipf.get('r_squared', 0)
        if r_squared > 0.9:
            report_lines.append("✅ Strong Zipfian distribution")
        elif r_squared > 0.7:
            report_lines.append("⚠️  Moderate Zipfian distribution")
        else:
            report_lines.append("❌ Weak Zipfian distribution")
    
    # Heaps law analysis
    if 'heaps_analysis' in results:
        heaps = results['heaps_analysis']
        report_lines.append("\n📈 HEAPS' LAW ANALYSIS")
        report_lines.append("-" * 40)
        report_lines.append(f"K parameter:...................... {heaps.get('K', 0):.3f}")
        report_lines.append(f"β parameter:...................... {heaps.get('beta', 0):.3f}")
        report_lines.append(f"R-squared:........................ {heaps.get('r_squared', 0):.3f}")
        
        if 'confidence_lower' in heaps and 'confidence_upper' in heaps:
            report_lines.append(f"95% CI for β:..................... "
                              f"[{heaps['confidence_lower']:.3f}, {heaps['confidence_upper']:.3f}]")
        
        # Interpretation
        beta = heaps.get('beta', 0)
        if 0.4 <= beta <= 0.6:
            report_lines.append("✅ Typical vocabulary growth pattern")
        elif beta < 0.4:
            report_lines.append("⚠️  Slow vocabulary growth")
        else:
            report_lines.append("⚠️  Fast vocabulary growth")
    
    # Advanced metrics
    if 'advanced_metrics' in results:
        metrics = results['advanced_metrics']
        report_lines.append("\n🎯 ADVANCED LINGUISTIC METRICS")
        report_lines.append("-" * 40)
        
        key_metrics = [
            ('guiraud_r', 'Guiraud\'s R'),
            ('herdan_c', 'Herdan\'s C'),
            ('yule_k', 'Yule\'s K'),
            ('shannon_entropy', 'Shannon Entropy'),
            ('hapax_legomena_ratio', 'Hapax Legomena Ratio')
        ]
        
        for key, label in key_metrics:
            if key in metrics:
                report_lines.append(f"{label:.<30} {metrics[key]:.3f}")
    
    # Warnings and notes
    report_lines.append("\n⚠️  INTERPRETATION NOTES")
    report_lines.append("-" * 40)
    report_lines.append("• Zipf and Heaps law analyses are approximations")
    report_lines.append("• Small corpora may produce unreliable estimates")
    report_lines.append("• Results should be interpreted in linguistic context")
    report_lines.append("• Confidence intervals indicate estimation uncertainty")
    
    report_lines.append("\n" + "=" * 80)
    
    return "\n".join(report_lines)