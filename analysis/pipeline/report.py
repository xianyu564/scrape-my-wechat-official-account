#!/usr/bin/env python3
"""
Markdown report generation with cheer-up tone and scientific structure
"""

import os
import json
from datetime import datetime
from typing import Dict, List, Any, Counter as CounterType
from collections import Counter
import pandas as pd


def write_report(output_dir: str,
                corpus_stats: Dict[str, Any],
                freq_overall: Counter,
                freq_by_year: Dict[str, Counter],
                tfidf_results: pd.DataFrame,
                zipf_results: Dict[str, float],
                heaps_results: Dict[str, float],
                lexical_metrics: Dict[str, float],
                ngram_stats: Dict[str, int],
                growth_data: List[Dict[str, Any]],
                analysis_params: Dict[str, Any],
                tokenizer_info: Dict[str, Any]) -> str:
    """
    Generate comprehensive markdown report with cheer-up tone
    
    Args:
        output_dir: Output directory
        corpus_stats: Corpus statistics
        freq_overall: Overall frequencies
        freq_by_year: Frequencies by year
        tfidf_results: TF-IDF analysis results
        zipf_results: Zipf law analysis
        heaps_results: Heaps law analysis
        lexical_metrics: Lexical diversity metrics
        ngram_stats: N-gram statistics
        growth_data: Year-over-year growth data
        analysis_params: Analysis parameters
        tokenizer_info: Tokenizer information
        
    Returns:
        str: Report file path
    """
    report_path = os.path.join(output_dir, "report.md")
    
    with open(report_path, 'w', encoding='utf-8') as f:
        # Write header
        _write_header(f, corpus_stats, analysis_params)
        
        # Write executive summary
        _write_executive_summary(f, corpus_stats, lexical_metrics, ngram_stats, growth_data)
        
        # Write methods section
        _write_methods(f, tokenizer_info, analysis_params)
        
        # Write global overview
        _write_global_overview(f, corpus_stats, lexical_metrics, zipf_results, heaps_results)
        
        # Write yearly snapshots
        _write_yearly_snapshots(f, freq_by_year, tfidf_results)
        
        # Write phrase inventory
        _write_phrase_inventory(f, ngram_stats, freq_overall)
        
        # Write YoY movers
        _write_yoy_movers(f, growth_data)
        
        # Write reproduction notes
        _write_repro_notes(f, analysis_params, tokenizer_info)
    
    print(f"📝 Report generated: {report_path}")
    return report_path


def _write_header(f, corpus_stats: Dict[str, Any], analysis_params: Dict[str, Any]) -> None:
    """Write report header"""
    f.write("# Corpus Linguistic Report (Cheer-Up Edition)\n\n")
    f.write("> 🌟 **A journey through words that tell my story** 🌟\n\n")
    f.write("*This analysis celebrates the evolution of thoughts, ideas, and expressions across time.*\n\n")
    f.write("---\n\n")


def _write_executive_summary(f, 
                           corpus_stats: Dict[str, Any], 
                           lexical_metrics: Dict[str, float],
                           ngram_stats: Dict[str, int],
                           growth_data: List[Dict[str, Any]]) -> None:
    """Write executive summary with highlights"""
    f.write("## 1. Executive Summary\n\n")
    
    # Basic stats
    total_articles = corpus_stats.get('total_articles', 0)
    years = corpus_stats.get('years', [])
    year_range = f"{min(years)}–{max(years)}" if len(years) > 1 else str(years[0]) if years else "N/A"
    
    f.write(f"- **Articles**: {total_articles:,}\n")
    f.write(f"- **Years**: {year_range}\n")
    
    # N-gram detection
    max_n = max([n for n, count in ngram_stats.items() if isinstance(n, int) and count > 0], default=1)
    f.write(f"- **Detected n-gram lengths**: 1–{max_n}\n")
    
    # Highlights
    if growth_data:
        top_growth = sorted(growth_data, key=lambda x: x.get('growth', 0), reverse=True)[:3]
        growth_words = [item['word'] for item in top_growth if item.get('growth', 0) > 0]
        if growth_words:
            f.write(f"- **Highlights**: Rising themes include {', '.join(growth_words[:2])} · ")
        else:
            f.write("- **Highlights**: Steady linguistic evolution · ")
    else:
        f.write("- **Highlights**: Rich vocabulary diversity · ")
    
    # Cheerful sentiment
    vocab_diversity = lexical_metrics.get('ttr', 0)
    if vocab_diversity > 0.15:
        mood = "vibrant linguistic creativity"
    elif vocab_diversity > 0.10:
        mood = "thoughtful expression depth"
    else:
        mood = "focused thematic consistency"
    
    f.write(f"Years of {mood} captured in words 🎨\n\n")


def _write_methods(f, tokenizer_info: Dict[str, Any], analysis_params: Dict[str, Any]) -> None:
    """Write methods section"""
    f.write("## 2. Methods\n\n")
    
    tokenizer_name = tokenizer_info.get('name', 'unknown')
    fallback_note = " (fallback: jieba)" if tokenizer_name == 'jieba' else ""
    
    f.write(f"- **Tokenizer**: {tokenizer_name}{fallback_note}; phrase merging via PMI/log-likelihood\n")
    f.write("- **TF-IDF**: via scikit-learn; pre-tokenized input\n")
    f.write("- **Word clouds**: via wordcloud with CJK font support\n")
    f.write("- **Laws checked**: Zipf's law (rank-frequency), Heaps' law (vocabulary growth)\n\n")


def _write_global_overview(f,
                         corpus_stats: Dict[str, Any],
                         lexical_metrics: Dict[str, float],
                         zipf_results: Dict[str, float],
                         heaps_results: Dict[str, float]) -> None:
    """Write global overview with metrics table"""
    f.write("## 3. Global Overview\n\n")
    
    # Metrics table
    f.write("| Metric | Value |\n")
    f.write("|---|---:|\n")
    f.write(f"| Articles | {corpus_stats.get('total_articles', 0):,} |\n")
    f.write(f"| Tokens (after merge) | {lexical_metrics.get('total_tokens', 0):,} |\n")
    f.write(f"| Vocab size | {lexical_metrics.get('unique_tokens', 0):,} |\n")
    f.write(f"| TTR | {lexical_metrics.get('ttr', 0):.3f} |\n")
    
    K = heaps_results.get('K', 0)
    beta = heaps_results.get('beta', 0)
    f.write(f"| Heaps K / β | {K:.1f} / {beta:.3f} |\n")
    
    slope = zipf_results.get('slope', 0)
    r_squared = zipf_results.get('r_squared', 0)
    f.write(f"| Zipf slope (R²) | {slope:.3f} ({r_squared:.3f}) |\n\n")
    
    # Add visualization references
    f.write("![Zipf panels](out/fig_zipf_panels.png)\n\n")
    f.write("![Heaps law](out/fig_heaps.png)\n\n")


def _write_yearly_snapshots(f, 
                          freq_by_year: Dict[str, Counter],
                          tfidf_results: pd.DataFrame) -> None:
    """Write yearly snapshots with word clouds"""
    f.write("## 4. Yearly Snapshots\n\n")
    
    years = sorted(freq_by_year.keys())
    
    for year in years:
        f.write(f"### {year}\n\n")
        
        # Top 20 frequency table
        f.write("**Top 20 (frequency)**\n\n")
        f.write("| Rank | Token | Freq |\n")
        f.write("|---:|---|---:|\n")
        
        year_freq = freq_by_year[year]
        for rank, (word, freq) in enumerate(year_freq.most_common(20), 1):
            f.write(f"| {rank} | {word} | {freq:,} |\n")
        
        f.write("\n")
        
        # Top 20 TF-IDF table
        f.write("**Top 20 (TF-IDF)**\n\n")
        f.write("| Rank | Token | Score |\n") 
        f.write("|---:|---|---:|\n")
        
        if not tfidf_results.empty and 'year' in tfidf_results.columns:
            year_tfidf = tfidf_results[tfidf_results['year'] == year].head(20)
            for rank, (_, row) in enumerate(year_tfidf.iterrows(), 1):
                word = row['word']
                score = row['score']
                f.write(f"| {rank} | {word} | {score:.4f} |\n")
        else:
            f.write("| - | Analysis pending | - |\n")
        
        f.write("\n")
        
        # Word cloud
        f.write("**Word Cloud**\n\n")
        f.write(f"![{year} cloud](out/cloud_{year}.png)\n\n")


def _write_phrase_inventory(f, ngram_stats: Dict[str, int], freq_overall: Counter) -> None:
    """Write phrase inventory section"""
    f.write("## 5. Phrase Inventory\n\n")
    
    # N-gram lengths detected
    detected_lengths = [n for n, count in ngram_stats.items() if isinstance(n, int) and count > 0]
    if detected_lengths:
        max_n = max(detected_lengths)
        f.write(f"- **N-gram lengths detected**: 1–{max_n}\n\n")
    else:
        f.write("- **N-gram lengths detected**: 1 (unigrams only)\n\n")
    
    # Sample phrases table
    f.write("**Samples**\n\n")
    f.write("| n | Example phrases (≤10) |\n")
    f.write("|---:|---|\n")
    
    # Extract sample phrases by length
    samples_by_length = _extract_phrase_samples(freq_overall)
    
    for n in range(1, 9):  # Up to 8-grams
        if n in samples_by_length and samples_by_length[n]:
            examples = ", ".join(samples_by_length[n][:10])
            f.write(f"| {n} | {examples} |\n")
        else:
            f.write(f"| {n} | - |\n")
    
    f.write("\n")


def _write_yoy_movers(f, growth_data: List[Dict[str, Any]]) -> None:
    """Write year-over-year movers section"""
    f.write("## 6. YoY Movers (Top 20)\n\n")
    
    if not growth_data:
        f.write("*No year-over-year data available (single year corpus)*\n\n")
        return
    
    f.write("| Token | Year(t-1) | Year(t) | Δ |\n")
    f.write("|---|---:|---:|---:|\n")
    
    # Sort by growth and take top 20
    sorted_growth = sorted(growth_data, key=lambda x: x.get('growth', 0), reverse=True)[:20]
    
    for item in sorted_growth:
        word = item.get('word', '')
        prev_count = item.get('prev_count', 0)
        curr_count = item.get('curr_count', 0)
        growth = item.get('growth', 0)
        
        delta_str = f"+{growth}" if growth > 0 else str(growth)
        f.write(f"| {word} | {prev_count} | {curr_count} | {delta_str} |\n")
    
    f.write("\n")


def _write_repro_notes(f, analysis_params: Dict[str, Any], tokenizer_info: Dict[str, Any]) -> None:
    """Write reproduction notes"""
    f.write("## 7. Notes for Repro & Parameters\n\n")
    
    max_n = analysis_params.get('max_n', 6)
    min_freq = analysis_params.get('min_freq', 5)
    collocation = analysis_params.get('collocation', 'pmi')
    
    f.write(f"- `MAX_N={max_n}, MIN_FREQ={min_freq}, COLLOCATION='{collocation}'`\n")
    f.write(f"- `TOKENIZER='{tokenizer_info.get('name', 'unknown')}'`\n")
    f.write("- `RUN_ANALYSIS=True, RUN_VISUALIZATION=True`\n")
    f.write("- All results reproducible with `SEED=42`\n\n")
    
    # Generation timestamp
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    f.write("---\n\n")
    f.write(f"*Report generated: {timestamp}*\n")
    f.write("*Analysis engine: Robust Chinese + Mixed-Language Linguistic Analysis System*\n")
    f.write("*Designed to cheer myself up from my past papers ✨*\n")


def _extract_phrase_samples(frequencies: Counter) -> Dict[int, List[str]]:
    """Extract sample phrases by n-gram length"""
    samples = {}
    
    for phrase, freq in frequencies.most_common(1000):  # Look at top frequent terms
        # Determine phrase length
        if '_' in phrase:
            # Compound n-gram
            parts = phrase.split('_')
            n = len(parts)
        else:
            # Single token
            n = 1
        
        if n not in samples:
            samples[n] = []
        
        if len(samples[n]) < 15:  # Collect up to 15 examples per length
            samples[n].append(phrase)
    
    return samples