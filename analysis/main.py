#!/usr/bin/env python3
"""
Robust Chinese + Mixed-Language Linguistic Analysis
Two-phase design: (1) theory-only analysis, (2) presentation
All knobs live here in main.py
"""

import argparse
import json
import os
import pickle
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

# Add pipeline to path
sys.path.insert(0, str(Path(__file__).parent / "pipeline"))

# Import pipeline modules
from corpus_io import get_corpus_stats, load_corpus, read_text, split_by_year
from ngrams import build_ngrams, get_ngram_stats
from report import write_report
from stats import (
    analyze_heaps_law,
    analyze_zipf_law,
    calculate_frequencies,
    calculate_frequencies_by_year,
    calculate_lexical_metrics,
    calculate_tfidf,
    get_year_over_year_growth,
    save_summary_stats,
)
from tokenizer import MixedLanguageTokenizer
from viz import (
    create_growth_chart,
    create_heaps_plot,
    create_wordcloud,
    create_yearly_comparison_chart,
    create_zipf_panels,
)


def print_and_save_config(config: Dict[str, Any], output_dir: str) -> None:
    """
    Print configuration to console and save to summary.json
    """
    print("=" * 70)
    print("🔧 FINAL CONFIGURATION PARAMETERS")
    print("=" * 70)

    # Print configuration
    for section, params in config.items():
        print(f"\n📋 {section.upper()}:")
        if isinstance(params, dict):
            for key, value in params.items():
                print(f"  {key}: {value}")
        else:
            print(f"  {params}")

    print("=" * 70)

    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # Save to summary.json
    summary_path = os.path.join(output_dir, "summary.json")
    summary_data = {"config": config}

    # Load existing summary if it exists
    if os.path.exists(summary_path):
        try:
            with open(summary_path, encoding='utf-8') as f:
                existing_data = json.load(f)
            existing_data["config"] = config
            summary_data = existing_data
        except Exception:
            pass  # Use new summary data if loading fails

    # Save updated summary
    with open(summary_path, 'w', encoding='utf-8') as f:
        json.dump(summary_data, f, ensure_ascii=False, indent=2)

    print(f"💾 Configuration saved to: {summary_path}")


def parse_arguments() -> argparse.Namespace:
    """
    Parse command line arguments
    """
    parser = argparse.ArgumentParser(
        description="Robust Chinese + Mixed-Language Linguistic Analysis",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python analysis/main.py --max-n 8 --collocation llr
  python analysis/main.py --corpus ../Wechat-Backup --years 2022,2023
  python analysis/main.py --tokenizer jieba --min-freq 3
        """
    )

    # Execution control
    parser.add_argument('--analysis', action='store_true', default=None,
                       help='Run analysis phase (default: True)')
    parser.add_argument('--no-analysis', dest='analysis', action='store_false',
                       help='Skip analysis phase')
    parser.add_argument('--visualization', action='store_true', default=None,
                       help='Run visualization phase (default: True)')
    parser.add_argument('--no-visualization', dest='visualization', action='store_false',
                       help='Skip visualization phase')

    # Data paths
    parser.add_argument('--corpus', type=str,
                       help='Corpus root directory')
    parser.add_argument('--output', type=str,
                       help='Output directory')

    # Analysis parameters
    parser.add_argument('--max-n', type=int,
                       help='Maximum n-gram length (default: 8)')
    parser.add_argument('--min-freq', type=int,
                       help='Minimum frequency for n-gram retention (default: 5)')
    parser.add_argument('--collocation', choices=['pmi', 'llr'],
                       help='Collocation method: pmi or llr (default: pmi)')
    parser.add_argument('--pmi-threshold', type=float,
                       help='PMI threshold for phrase validation (default: 3.0)')
    parser.add_argument('--llr-threshold', type=float,
                       help='LLR threshold for phrase validation (default: 10.83)')

    # Tokenizer parameters
    parser.add_argument('--tokenizer', choices=['auto', 'pkuseg', 'jieba'],
                       help='Tokenizer type (default: auto)')

    # Time filtering
    parser.add_argument('--start-date', type=str,
                       help='Start date filter (YYYY-MM-DD)')
    parser.add_argument('--end-date', type=str,
                       help='End date filter (YYYY-MM-DD)')
    parser.add_argument('--years', type=str,
                       help='Year filter (comma-separated, e.g., 2021,2022)')

    # Visualization parameters
    parser.add_argument('--font-path', type=str,
                       help='Path to Chinese font file')
    parser.add_argument('--color-scheme', choices=['nature', 'science', 'calm'],
                       help='Color scheme for visualizations (default: nature)')

    # Reproducibility
    parser.add_argument('--seed', type=int,
                       help='Random seed for reproducible results (default: 42)')

    return parser.parse_args()


def run_analysis(
    corpus_root: str,
    output_dir: str,
    # Analysis parameters
    max_n: int = 8,
    min_freq: int = 5,
    collocation: str = 'pmi',
    pmi_threshold: float = 3.0,
    llr_threshold: float = 10.83,
    # TF-IDF parameters
    tfidf_min_df: int = 1,
    tfidf_max_df: float = 0.98,
    tfidf_topk: int = 150,
    # Tokenizer parameters
    tokenizer_type: str = "auto",
    # Time filtering
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    years: Optional[List[str]] = None,
    # Data paths
    stopwords_zh_path: str = "data/stopwords.zh.txt",
    stopwords_en_path: str = "data/stopwords.en.txt",
    allow_singletons_path: str = "data/allow_singletons.zh.txt",
    # Reproducibility
    seed: int = 42
) -> Dict[str, Any]:
    """
    Phase 1: Theory-only analysis
    Process corpus and generate all statistics without visualization
    
    Returns:
        Dict: Complete analysis results
    """
    print("=" * 70)
    print("🧠 PHASE 1: THEORY-ONLY ANALYSIS")
    print("=" * 70)

    # Set random seed for reproducibility
    import random

    import numpy as np
    random.seed(seed)
    np.random.seed(seed)

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    # Load corpus
    print(f"📁 Loading corpus from: {corpus_root}")
    articles = load_corpus(
        root_dir=corpus_root,
        start_date=start_date,
        end_date=end_date,
        years=years
    )

    if not articles:
        raise ValueError("❌ No articles found in corpus")

    corpus_stats = get_corpus_stats(articles)
    print(f"📊 Corpus loaded: {corpus_stats}")

    # Initialize tokenizer
    print(f"🔤 Initializing tokenizer: {tokenizer_type}")

    # Prepare user dictionary paths
    extra_user_dicts = []
    default_user_dict = "data/user_dict.zh.txt"
    tech_terms_dict = "data/tech_terms.txt"

    if os.path.exists(default_user_dict):
        extra_user_dicts.append(default_user_dict)
    if os.path.exists(tech_terms_dict):
        extra_user_dicts.append(tech_terms_dict)

    tokenizer = MixedLanguageTokenizer(
        tokenizer_type=tokenizer_type,
        extra_user_dicts=extra_user_dicts,
        stopwords_zh_path=stopwords_zh_path,
        stopwords_en_path=stopwords_en_path,
        allow_singletons_path=allow_singletons_path
    )

    # Tokenize all articles
    print("📝 Tokenizing articles...")
    all_tokens = []
    articles_by_year = split_by_year(articles)
    tokens_by_year = {}
    texts_by_year = {}

    for year, year_articles in articles_by_year.items():
        year_tokens = []
        year_texts = []

        for article in year_articles:
            text = read_text(article)
            if text:
                tokens = tokenizer.tokenize(text)
                if tokens:
                    all_tokens.append(tokens)
                    year_tokens.append(tokens)
                    year_texts.append(text)

        if year_tokens:
            tokens_by_year[year] = year_tokens
            texts_by_year[year] = year_texts

    print(f"✅ Tokenized {len(all_tokens)} articles across {len(tokens_by_year)} years")

    # Build variable-length n-grams
    print(f"🔢 Building n-grams: max_n={max_n}, collocation={collocation}")

    # Flatten tokens for n-gram analysis
    flat_tokens = []
    for tokens in all_tokens:
        flat_tokens.extend(tokens)

    merged_tokens, ngram_counts = build_ngrams(
        tokens=flat_tokens,
        max_n=max_n,
        min_freq=min_freq,
        collocation=collocation,
        pmi_threshold=pmi_threshold,
        llr_threshold=llr_threshold
    )

    # Rebuild corpus with merged tokens
    merged_corpus = []
    merged_by_year = {}

    start_idx = 0
    for i, original_tokens in enumerate(all_tokens):
        doc_length = len(original_tokens)
        # Find corresponding article year
        doc_year = None
        for year, year_tokens in tokens_by_year.items():
            if i < len(year_tokens):
                doc_year = year
                break

        # Extract merged tokens for this document (approximation)
        end_idx = start_idx + doc_length
        doc_merged = merged_tokens[start_idx:end_idx] if end_idx <= len(merged_tokens) else merged_tokens[start_idx:]
        merged_corpus.append(doc_merged)

        if doc_year:
            if doc_year not in merged_by_year:
                merged_by_year[doc_year] = []
            merged_by_year[doc_year].append(doc_merged)

        start_idx = end_idx

    # Calculate statistics
    print("📊 Calculating frequency statistics...")
    freq_overall = calculate_frequencies(merged_corpus)
    freq_by_year = calculate_frequencies_by_year(merged_by_year)

    print("🔍 Calculating TF-IDF...")
    tfidf_results = calculate_tfidf(
        texts_by_year=texts_by_year,
        tokenizer_func=tokenizer.tokenize,
        min_df=tfidf_min_df,
        max_df=tfidf_max_df,
        topk=tfidf_topk
    )

    print("📈 Analyzing Zipf's law...")
    zipf_results = analyze_zipf_law(freq_overall)

    print("📊 Analyzing Heaps' law...")
    heaps_results = analyze_heaps_law(merged_corpus)

    print("🔤 Calculating lexical metrics...")
    lexical_metrics = calculate_lexical_metrics(
        merged_corpus,
        stopwords_zh=tokenizer.stopwords_zh if hasattr(tokenizer, 'stopwords_zh') else None,
        stopwords_en=tokenizer.stopwords_en if hasattr(tokenizer, 'stopwords_en') else None
    )

    print("📅 Analyzing year-over-year growth...")
    growth_data = get_year_over_year_growth(freq_by_year, topk=20)

    print("🔢 Collecting n-gram statistics...")
    ngram_stats = get_ngram_stats(merged_tokens)
    ngram_stats.update(ngram_counts)  # Add counts by length

    # Validate n-gram detection
    detected_lengths = [n for n in range(1, max_n + 1) if ngram_counts.get(n, 0) > 0]
    if not detected_lengths:
        print("⚠️  WARNING: No n-grams detected, consider lowering min_freq")
    else:
        print(f"✅ N-gram lengths detected: {detected_lengths}")

    # Prepare results
    analysis_results = {
        'corpus_stats': corpus_stats,
        'freq_overall': freq_overall,
        'freq_by_year': freq_by_year,
        'tfidf_results': tfidf_results,
        'zipf_results': zipf_results,
        'heaps_results': heaps_results,
        'lexical_metrics': lexical_metrics,
        'growth_data': growth_data,
        'ngram_stats': ngram_stats,
        'merged_tokens': merged_tokens,
        'merged_corpus': merged_corpus,
        'tokenizer_info': tokenizer.get_tokenizer_info(),
        'analysis_params': {
            'max_n': max_n,
            'min_freq': min_freq,
            'collocation': collocation,
            'pmi_threshold': pmi_threshold,
            'llr_threshold': llr_threshold,
            'tfidf_min_df': tfidf_min_df,
            'tfidf_max_df': tfidf_max_df,
            'tokenizer_type': tokenizer_type,
            'seed': seed
        }
    }

    # Save summary statistics
    summary_path = os.path.join(output_dir, "summary.json")
    save_summary_stats(
        freq_overall=freq_overall,
        freq_by_year=freq_by_year,
        tfidf_results=tfidf_results,
        zipf_results=zipf_results,
        heaps_results=heaps_results,
        lexical_metrics=lexical_metrics,
        ngram_stats=ngram_stats,
        output_path=summary_path
    )

    # Save complete results for phase 2
    results_path = os.path.join(output_dir, "analysis_results.pkl")
    with open(results_path, 'wb') as f:
        pickle.dump(analysis_results, f)

    print(f"💾 Analysis results saved to: {results_path}")
    print("✅ Phase 1 complete!\n")

    return analysis_results


def run_presentation(
    output_dir: str,
    # Visualization parameters
    generate_wordclouds: bool = True,
    generate_scientific_plots: bool = True,
    generate_report: bool = True,
    # Word cloud parameters
    wordcloud_max_words: int = 200,
    yearly_wordcloud_max_words: int = 100,
    # Font and styling
    font_path: Optional[str] = None,
    color_scheme: str = "nature",
    # Chart parameters
    yearly_comparison_top_n: int = 20,
    growth_chart_top_n: int = 20
) -> Dict[str, str]:
    """
    Phase 2: Presentation
    Generate beautiful visualizations and reports from analysis results
    
    Returns:
        Dict: Generated file paths
    """
    print("=" * 70)
    print("🎨 PHASE 2: PRESENTATION")
    print("=" * 70)

    # Load analysis results
    results_path = os.path.join(output_dir, "analysis_results.pkl")
    if not os.path.exists(results_path):
        raise FileNotFoundError(f"❌ Analysis results not found: {results_path}")

    print("📂 Loading analysis results...")
    with open(results_path, 'rb') as f:
        results = pickle.load(f)

    generated_files = {}

    # Generate scientific plots
    if generate_scientific_plots:
        print("📈 Generating scientific plots...")

        # Zipf law panels
        zipf_path = os.path.join(output_dir, "fig_zipf_panels.png")
        create_zipf_panels(
            frequencies=results['freq_overall'],
            output_path=zipf_path,
            zipf_results=results['zipf_results'],
            font_path=font_path,
            color_scheme=color_scheme
        )
        generated_files['zipf_panels'] = zipf_path

        # Heaps law plot
        heaps_path = os.path.join(output_dir, "fig_heaps.png")
        create_heaps_plot(
            corpus_tokens=results['merged_corpus'],
            output_path=heaps_path,
            heaps_results=results['heaps_results'],
            font_path=font_path,
            color_scheme=color_scheme
        )
        generated_files['heaps_plot'] = heaps_path

        # Yearly comparison chart
        yearly_comparison_path = os.path.join(output_dir, "fig_yearly_comparison.png")
        create_yearly_comparison_chart(
            freq_by_year=results['freq_by_year'],
            output_path=yearly_comparison_path,
            top_n=yearly_comparison_top_n,
            font_path=font_path,
            color_scheme=color_scheme
        )
        generated_files['yearly_comparison'] = yearly_comparison_path

        # Growth chart
        if results['growth_data']:
            growth_path = os.path.join(output_dir, "fig_growth.png")
            create_growth_chart(
                growth_data=results['growth_data'],
                output_path=growth_path,
                top_n=growth_chart_top_n,
                font_path=font_path,
                color_scheme=color_scheme
            )
            generated_files['growth_chart'] = growth_path

    # Generate word clouds
    if generate_wordclouds:
        print("🎨 Generating word clouds...")

        # Overall word cloud
        overall_cloud_path = os.path.join(output_dir, "cloud_overall.png")
        create_wordcloud(
            frequencies=results['freq_overall'],
            output_path=overall_cloud_path,
            title="Overall Word Cloud",
            font_path=font_path,
            max_words=wordcloud_max_words,
            color_scheme=color_scheme
        )
        generated_files['cloud_overall'] = overall_cloud_path

        # Yearly word clouds
        yearly_clouds = []
        for year, freq in results['freq_by_year'].items():
            cloud_path = os.path.join(output_dir, f"cloud_{year}.png")
            create_wordcloud(
                frequencies=freq,
                output_path=cloud_path,
                title=f"{year} Word Cloud",
                font_path=font_path,
                max_words=yearly_wordcloud_max_words,
                color_scheme=color_scheme
            )
            yearly_clouds.append(cloud_path)

        generated_files['yearly_clouds'] = yearly_clouds

    # Generate report
    if generate_report:
        print("📝 Generating markdown report...")
        report_path = write_report(
            output_dir=output_dir,
            corpus_stats=results['corpus_stats'],
            freq_overall=results['freq_overall'],
            freq_by_year=results['freq_by_year'],
            tfidf_results=results['tfidf_results'],
            zipf_results=results['zipf_results'],
            heaps_results=results['heaps_results'],
            lexical_metrics=results['lexical_metrics'],
            ngram_stats=results['ngram_stats'],
            growth_data=results['growth_data'],
            analysis_params=results['analysis_params'],
            tokenizer_info=results['tokenizer_info']
        )
        generated_files['report'] = report_path

    print("✅ Phase 2 complete!")
    print("\n📁 Generated files:")
    for file_type, path in generated_files.items():
        if isinstance(path, list):
            print(f"  - {file_type}: {len(path)} files")
        else:
            print(f"  - {file_type}: {os.path.basename(path)}")

    return generated_files


def main():
    """
    Main function with all configuration knobs
    """
    # Parse command line arguments
    args = parse_arguments()

    # =================================================================
    # 🔧 CONFIGURATION PARAMETERS
    # =================================================================

    # === Execution Control ===
    RUN_ANALYSIS = args.analysis if args.analysis is not None else True
    RUN_VISUALIZATION = args.visualization if args.visualization is not None else True

    # === Data Paths ===
    CORPUS_ROOT = args.corpus or "../Wechat-Backup/文不加点的张衔瑜"
    OUTPUT_DIR = args.output or "out"

    # === N-gram & Collocation Parameters ===
    MAX_N = args.max_n or 8
    MIN_FREQ = args.min_freq or 5
    COLLOCATION = args.collocation or "pmi"
    PMI_THRESHOLD = args.pmi_threshold or 3.0
    LLR_THRESHOLD = args.llr_threshold or 10.83

    # === Tokenization Parameters ===
    TOKENIZER_TYPE = args.tokenizer or "auto"
    STOPWORDS_ZH_PATH = "data/stopwords.zh.txt"
    STOPWORDS_EN_PATH = "data/stopwords.en.txt"
    ALLOW_SINGLETONS_PATH = "data/allow_singletons.zh.txt"

    # === TF-IDF Parameters ===
    TFIDF_MIN_DF = 1
    TFIDF_MAX_DF = 0.98
    TFIDF_TOPK = 150

    # === Time Filtering ===
    START_DATE = args.start_date
    END_DATE = args.end_date
    YEARS = args.years.split(',') if args.years else None

    # === Visualization Parameters ===
    GENERATE_WORDCLOUDS = True
    GENERATE_SCIENTIFIC_PLOTS = True
    GENERATE_REPORT = True

    WORDCLOUD_MAX_WORDS = 200
    YEARLY_WORDCLOUD_MAX_WORDS = 100

    FONT_PATH = args.font_path
    COLOR_SCHEME = args.color_scheme or "nature"

    YEARLY_COMPARISON_TOP_N = 20
    GROWTH_CHART_TOP_N = 20

    # === Reproducibility ===
    SEED = args.seed or 42

    # =================================================================
    # 📋 PRINT AND SAVE FINAL CONFIGURATION
    # =================================================================

    config = {
        "execution": {
            "run_analysis": RUN_ANALYSIS,
            "run_visualization": RUN_VISUALIZATION
        },
        "data_paths": {
            "corpus_root": CORPUS_ROOT,
            "output_dir": OUTPUT_DIR,
            "stopwords_zh_path": STOPWORDS_ZH_PATH,
            "stopwords_en_path": STOPWORDS_EN_PATH,
            "allow_singletons_path": ALLOW_SINGLETONS_PATH
        },
        "analysis": {
            "max_n": MAX_N,
            "min_freq": MIN_FREQ,
            "collocation": COLLOCATION,
            "pmi_threshold": PMI_THRESHOLD,
            "llr_threshold": LLR_THRESHOLD,
            "tokenizer_type": TOKENIZER_TYPE
        },
        "tfidf": {
            "min_df": TFIDF_MIN_DF,
            "max_df": TFIDF_MAX_DF,
            "topk": TFIDF_TOPK
        },
        "time_filtering": {
            "start_date": START_DATE,
            "end_date": END_DATE,
            "years": YEARS
        },
        "visualization": {
            "generate_wordclouds": GENERATE_WORDCLOUDS,
            "generate_scientific_plots": GENERATE_SCIENTIFIC_PLOTS,
            "generate_report": GENERATE_REPORT,
            "wordcloud_max_words": WORDCLOUD_MAX_WORDS,
            "yearly_wordcloud_max_words": YEARLY_WORDCLOUD_MAX_WORDS,
            "font_path": FONT_PATH,
            "color_scheme": COLOR_SCHEME,
            "yearly_comparison_top_n": YEARLY_COMPARISON_TOP_N,
            "growth_chart_top_n": GROWTH_CHART_TOP_N
        },
        "reproducibility": {
            "seed": SEED
        }
    }

    print_and_save_config(config, OUTPUT_DIR)

    # =================================================================
    # 🚀 EXECUTION
    # =================================================================

    try:
        # Phase 1: Analysis
        if RUN_ANALYSIS:
            _ = run_analysis(
                corpus_root=CORPUS_ROOT,
                output_dir=OUTPUT_DIR,
                max_n=MAX_N,
                min_freq=MIN_FREQ,
                collocation=COLLOCATION,
                pmi_threshold=PMI_THRESHOLD,
                llr_threshold=LLR_THRESHOLD,
                tfidf_min_df=TFIDF_MIN_DF,
                tfidf_max_df=TFIDF_MAX_DF,
                tfidf_topk=TFIDF_TOPK,
                tokenizer_type=TOKENIZER_TYPE,
                start_date=START_DATE,
                end_date=END_DATE,
                years=YEARS,
                stopwords_zh_path=STOPWORDS_ZH_PATH,
                stopwords_en_path=STOPWORDS_EN_PATH,
                allow_singletons_path=ALLOW_SINGLETONS_PATH,
                seed=SEED
            )

            # Validation check
            summary_path = os.path.join(OUTPUT_DIR, "summary.json")
            with open(summary_path, encoding='utf-8') as f:
                summary = json.load(f)

            detected_lengths = summary.get('ngram_lengths_detected', [])
            if not detected_lengths or max(detected_lengths) < 2:
                print("⚠️  WARNING: Limited n-gram detection. Consider lowering MIN_FREQ.")
            else:
                print(f"✅ Self-check passed: n-gram lengths {detected_lengths}")

        # Phase 2: Visualization
        if RUN_VISUALIZATION:
            _ = run_presentation(
                output_dir=OUTPUT_DIR,
                generate_wordclouds=GENERATE_WORDCLOUDS,
                generate_scientific_plots=GENERATE_SCIENTIFIC_PLOTS,
                generate_report=GENERATE_REPORT,
                wordcloud_max_words=WORDCLOUD_MAX_WORDS,
                yearly_wordcloud_max_words=YEARLY_WORDCLOUD_MAX_WORDS,
                font_path=FONT_PATH,
                color_scheme=COLOR_SCHEME,
                yearly_comparison_top_n=YEARLY_COMPARISON_TOP_N,
                growth_chart_top_n=GROWTH_CHART_TOP_N
            )

        print("\n🎉 ANALYSIS COMPLETE!")
        print("=" * 70)

        if RUN_ANALYSIS and not RUN_VISUALIZATION:
            print("💡 Tip: Set --visualization to generate visuals")
        elif RUN_VISUALIZATION and not RUN_ANALYSIS:
            print("💡 Tip: Set --analysis to rerun analysis")

    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
