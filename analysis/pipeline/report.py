#!/usr/bin/env python3
"""
Enhanced Markdown report generation with Jinja2 templates and scientific structure
Task 6: 报告生成器升级
"""

import json
import os
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd

try:
    from jinja2 import Environment, FileSystemLoader, Template
    JINJA2_AVAILABLE = True
except ImportError:
    JINJA2_AVAILABLE = False
    print("⚠️  Jinja2 not available, using basic templating")


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
    Generate comprehensive markdown report with enhanced structure and templates
    
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

    # Initialize template environment if available
    template_env = None
    if JINJA2_AVAILABLE:
        template_dir = Path(__file__).parent.parent / "templates"
        if template_dir.exists():
            template_env = Environment(loader=FileSystemLoader(str(template_dir)))

    # Prepare data for templates
    report_data = _prepare_report_data(
        corpus_stats, freq_overall, freq_by_year, tfidf_results,
        zipf_results, heaps_results, lexical_metrics, ngram_stats,
        growth_data, analysis_params, tokenizer_info
    )

    with open(report_path, 'w', encoding='utf-8') as f:
        # Write header using template if available
        if template_env and JINJA2_AVAILABLE:
            _write_templated_header(f, template_env, report_data)
        else:
            _write_header(f, corpus_stats, analysis_params)

        # Write main sections
        _write_executive_summary(f, report_data)
        _write_methods(f, tokenizer_info, analysis_params)
        _write_global_overview(f, report_data)
        _write_yearly_snapshots(f, report_data)
        _write_phrase_inventory(f, report_data)
        _write_yoy_movers(f, growth_data)
        _write_statistical_laws(f, report_data)
        _write_cheer_up_summary(f, report_data)
        _write_repro_notes(f, analysis_params, tokenizer_info)

    print(f"📝 Enhanced report generated: {report_path}")
    return report_path


def _prepare_report_data(corpus_stats, freq_overall, freq_by_year, tfidf_results,
                        zipf_results, heaps_results, lexical_metrics, ngram_stats,
                        growth_data, analysis_params, tokenizer_info) -> Dict[str, Any]:
    """Prepare structured data for report templates"""

    # Basic corpus info
    total_articles = corpus_stats.get('total_articles', 0)
    years = corpus_stats.get('years', [])
    year_range = f"{min(years)}–{max(years)}" if len(years) > 1 else str(years[0]) if years else "N/A"

    # Prepare top terms
    top_terms_overall = [term for term, freq in freq_overall.most_common(10)]

    # Prepare yearly data
    yearly_data = {}
    for year, counter in freq_by_year.items():
        yearly_data[year] = {
            'total_terms': sum(counter.values()),
            'unique_terms': len(counter),
            'top_terms': [term for term, freq in counter.most_common(5)],
            'metrics': []
        }

    # Statistical laws summary
    laws_analysis = {
        'zipf': {
            'slope': zipf_results.get('slope', 0),
            'r_squared': zipf_results.get('r_squared', 0),
            'interpretation': _interpret_zipf_slope(zipf_results.get('slope', 0))
        },
        'heaps': {
            'beta': heaps_results.get('beta', 0),
            'K': heaps_results.get('K', 0),
            'r_squared': heaps_results.get('r_squared', 0),
            'interpretation': _interpret_heaps_beta(heaps_results.get('beta', 0))
        }
    }

    return {
        'generation_date': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        'title': 'Corpus Linguistic Analysis Report (Enhanced Edition)',
        'subtitle': 'A comprehensive journey through linguistic patterns and evolution',
        'description': 'This analysis reveals the hidden patterns, growth trends, and linguistic fingerprints of your written corpus.',
        'year_range': year_range,
        'total_articles': total_articles,
        'corpus_stats': corpus_stats,
        'top_terms_overall': top_terms_overall,
        'yearly_data': yearly_data,
        'lexical_metrics': lexical_metrics,
        'ngram_stats': ngram_stats,
        'laws_analysis': laws_analysis,
        'growth_data': growth_data,
        'analysis_params': analysis_params,
        'tokenizer_info': tokenizer_info
    }


def _write_templated_header(f, template_env, report_data):
    """Write header using Jinja2 template"""
    try:
        template = template_env.get_template('report_header.md.j2')
        header_content = template.render(**report_data)
        f.write(header_content)
        f.write("\n\n")
    except Exception as e:
        print(f"⚠️  Template rendering failed: {e}, using fallback")
        _write_header(f, report_data['corpus_stats'], report_data['analysis_params'])


def _interpret_zipf_slope(slope: float) -> str:
    """Interpret Zipf law slope"""
    if abs(slope + 1) < 0.2:
        return "Classical Zipfian distribution (ideal power law)"
    elif slope < -1.2:
        return "Steeper than classical (more concentrated vocabulary)"
    elif slope > -0.8:
        return "Flatter than classical (more distributed vocabulary)"
    else:
        return "Approximately Zipfian"


def _interpret_heaps_beta(beta: float) -> str:
    """Interpret Heaps law beta parameter"""
    if 0.4 <= beta <= 0.6:
        return "Classical sub-linear growth (typical for natural language)"
    elif beta < 0.4:
        return "Very slow vocabulary growth (repetitive text)"
    elif beta > 0.6:
        return "Fast vocabulary growth (diverse/technical text)"
    else:
        return "Non-standard growth pattern"


def _write_header(f, corpus_stats: Dict[str, Any], analysis_params: Dict[str, Any]) -> None:
    """Write report header (fallback when templates not available)"""
    f.write("# Corpus Linguistic Analysis Report (Enhanced Edition)\n\n")
    f.write("> 🌟 **A comprehensive journey through linguistic patterns and evolution** 🌟\n\n")
    f.write("*This analysis reveals the hidden patterns, growth trends, and linguistic fingerprints of your written corpus.*\n\n")
    f.write("---\n\n")


def _write_executive_summary(f, report_data: Dict[str, Any]) -> None:
    """Write enhanced executive summary"""
    f.write("## 1. Executive Summary\n\n")

    corpus_stats = report_data['corpus_stats']
    lexical_metrics = report_data['lexical_metrics']
    ngram_stats = report_data['ngram_stats']

    # Key highlights
    f.write("### 📊 Key Highlights\n\n")
    f.write(f"- **Corpus Size**: {corpus_stats.get('total_articles', 0):,} articles across {len(corpus_stats.get('years', []))} years\n")
    f.write(f"- **Vocabulary Richness**: {lexical_metrics.get('unique_tokens', 0):,} unique terms\n")
    f.write(f"- **Lexical Diversity (TTR)**: {lexical_metrics.get('ttr', 0):.3f}\n")

    # N-gram insights
    # Filter numeric values only (ignore dict values like 'ngram_stats_detailed')
    ngram_counts = {k: v for k, v in ngram_stats.items() if isinstance(v, (int, float))}
    total_ngrams = sum(ngram_counts.values()) if ngram_counts else 0
    if total_ngrams > 0:
        f.write(f"- **Multi-word Expressions**: {total_ngrams:,} n-grams detected\n")

    # Growth insights
    if report_data['growth_data']:
        growth_terms = len(report_data['growth_data'])
        f.write(f"- **Vocabulary Evolution**: {growth_terms} trending terms identified\n")

    f.write("\n### 🎯 Top Terms\n\n")
    for i, term in enumerate(report_data['top_terms_overall'][:5], 1):
        f.write(f"{i}. **{term}**\n")

    f.write("\n")


def _write_methods(f, tokenizer_info: Dict[str, Any], analysis_params: Dict[str, Any]) -> None:
    """Write methods section"""
    f.write("## 2. Methods & Configuration\n\n")

    tokenizer_name = tokenizer_info.get('name', 'unknown')
    fallback_note = " (fallback: jieba)" if 'jieba' in tokenizer_name.lower() else ""

    f.write("### 🔧 Processing Pipeline\n\n")
    f.write(f"- **Tokenizer**: {tokenizer_name}{fallback_note}\n")
    f.write(f"- **N-gram Analysis**: max_n={analysis_params.get('max_n', 8)}, collocation={analysis_params.get('collocation', 'pmi')}\n")
    f.write("- **TF-IDF**: scikit-learn with pre-tokenized input\n")
    f.write("- **Statistical Laws**: Zipf's law (rank-frequency), Heaps' law (vocabulary growth)\n")
    f.write(f"- **Random Seed**: {analysis_params.get('seed', 42)} (for reproducibility)\n\n")

    f.write("### 📚 User Dictionaries\n\n")
    user_dicts = tokenizer_info.get('user_dictionaries', 0)
    if user_dicts > 0:
        f.write(f"Loaded {user_dicts} custom dictionaries for technical term preservation.\n\n")
    else:
        f.write("No custom dictionaries loaded (using default tokenization).\n\n")


def _write_global_overview(f, report_data: Dict[str, Any]) -> None:
    """Write enhanced global overview"""
    f.write("## 3. Global Overview\n\n")

    corpus_stats = report_data['corpus_stats']
    lexical_metrics = report_data['lexical_metrics']

    f.write("### 📈 Corpus Characteristics\n\n")
    f.write("| Metric | Value |\n")
    f.write("|--------|-------|\n")
    f.write(f"| Total Articles | {corpus_stats.get('total_articles', 0):,} |\n")
    f.write(f"| Unique Terms | {lexical_metrics.get('unique_tokens', 0):,} |\n")
    f.write(f"| Total Tokens | {lexical_metrics.get('total_tokens', 0):,} |\n")
    f.write(f"| Type-Token Ratio | {lexical_metrics.get('ttr', 0):.4f} |\n")
    avg_tokens_per_doc = (
        lexical_metrics.get('total_tokens', 0) / corpus_stats.get('total_articles', 1)
        if corpus_stats.get('total_articles', 0) > 0 else 0
    )
    f.write(f"| Average Article Length | {avg_tokens_per_doc:.1f} tokens |\n")
    f.write("\n")


def _write_yearly_snapshots(f, report_data: Dict[str, Any]) -> None:
    """Write enhanced yearly snapshots"""
    f.write("## 4. Yearly Snapshots\n\n")

    yearly_data = report_data['yearly_data']

    for year, data in sorted(yearly_data.items()):
        f.write(f"### 📅 {year}\n\n")
        f.write(f"- **Total Terms**: {data['total_terms']:,}\n")
        f.write(f"- **Unique Terms**: {data['unique_terms']:,}\n")
        f.write(f"- **Top Terms**: {', '.join(data['top_terms'])}\n\n")


def _write_phrase_inventory(f, report_data: Dict[str, Any]) -> None:
    """Write phrase inventory section"""
    f.write("## 5. Phrase Inventory\n\n")

    ngram_stats = report_data['ngram_stats']

    # Show n-gram detection results
    for n in range(1, 9):
        count = ngram_stats.get(n, 0)
        if count > 0:
            f.write(f"- **{n}-grams**: {count:,} detected\n")

    f.write("\n")


def _write_yoy_movers(f, growth_data: List[Dict[str, Any]]) -> None:
    """Write year-over-year movers section"""
    f.write("## 6. Year-over-Year Movers\n\n")

    if not growth_data:
        f.write("No growth data available.\n\n")
        return

    # Sort by growth and show top movers
    sorted_growth = sorted(growth_data, key=lambda x: x.get('growth', 0), reverse=True)

    f.write("### 📈 Rising Terms\n\n")
    for item in sorted_growth[:10]:
        if item.get('growth', 0) > 0:
            f.write(f"- **{item['word']}**: +{item['growth']:.1f}% growth\n")

    f.write("\n")


def _write_statistical_laws(f, report_data: Dict[str, Any]) -> None:
    """Write statistical laws analysis section"""
    f.write("## 7. Statistical Laws Analysis\n\n")

    laws = report_data['laws_analysis']

    # Zipf's Law
    f.write("### 📈 Zipf's Law Analysis\n\n")
    zipf = laws['zipf']
    f.write(f"- **Slope**: {zipf['slope']:.3f}\n")
    f.write(f"- **R² Goodness of Fit**: {zipf['r_squared']:.3f}\n")
    f.write(f"- **Interpretation**: {zipf['interpretation']}\n\n")

    if zipf['r_squared'] > 0.8:
        f.write("✅ **Strong Zipfian behavior**: Your vocabulary follows a clear power-law distribution.\n\n")
    else:
        f.write("⚠️ **Moderate fit**: Some deviation from classical Zipfian distribution detected.\n\n")

    # Heaps' Law
    f.write("### 📊 Heaps' Law Analysis\n\n")
    heaps = laws['heaps']
    f.write(f"- **Growth Exponent (β)**: {heaps['beta']:.3f}\n")
    f.write(f"- **Scaling Constant (K)**: {heaps['K']:.3f}\n")
    f.write(f"- **R² Goodness of Fit**: {heaps['r_squared']:.3f}\n")
    f.write(f"- **Interpretation**: {heaps['interpretation']}\n\n")

    if 0.4 <= heaps['beta'] <= 0.6:
        f.write("✅ **Healthy vocabulary growth**: Your writing shows natural linguistic diversity expansion.\n\n")
    else:
        f.write("📋 **Unique growth pattern**: Your vocabulary development has distinctive characteristics.\n\n")


def _write_cheer_up_summary(f, report_data: Dict[str, Any]) -> None:
    """Write uplifting cheer-up summary section"""
    f.write("## 8. Cheer-Me-Up Summary 🌟\n\n")

    corpus_stats = report_data['corpus_stats']
    lexical_metrics = report_data['lexical_metrics']
    laws = report_data['laws_analysis']

    # Calculate years of writing
    years = corpus_stats.get('years', [])
    if len(years) > 1:
        year_ints = [int(year) for year in years if year.isdigit()]
        writing_span = max(year_ints) - min(year_ints) + 1 if year_ints else 1
    else:
        writing_span = 1
    

    f.write("Your linguistic journey is truly remarkable! Here's why:\n\n")

    # Productivity celebration
    articles_per_year = corpus_stats.get('total_articles', 0) / max(writing_span, 1)
    f.write(f"🚀 **Prolific Creator**: You've crafted {corpus_stats.get('total_articles', 0):,} articles over {writing_span} years ")
    f.write(f"(averaging {articles_per_year:.1f} pieces per year). Your dedication to expression is inspiring!\n\n")

    # Vocabulary richness
    vocab_size = lexical_metrics.get('total_unique_tokens', 0)
    if vocab_size > 10000:
        f.write(f"📚 **Rich Vocabulary**: With {vocab_size:,} unique terms, you command a vocabulary that rivals published authors. ")
        f.write("Your linguistic palette is vast and sophisticated!\n\n")
    elif vocab_size > 5000:
        f.write(f"📖 **Growing Lexicon**: Your {vocab_size:,} unique terms show impressive vocabulary development. ")
        f.write("You're building a distinctive voice!\n\n")
    else:
        f.write(f"🌱 **Focused Expression**: Your {vocab_size:,} terms are used with intention and clarity. ")
        f.write("Quality over quantity – a mark of thoughtful writing!\n\n")

    # Pattern recognition
    zipf_r2 = laws['zipf']['r_squared']
    if zipf_r2 > 0.9:
        f.write("🎯 **Natural Rhythm**: Your writing follows beautiful mathematical patterns (Zipf R² = ")
        f.write(f"{zipf_r2:.3f}), showing an intuitive grasp of linguistic flow.\n\n")

    # Growth mindset
    if len(years) > 1:
        f.write(f"📈 **Continuous Evolution**: Across {len(years)} years, your language continues to evolve and grow. ")
        f.write("You're not just writing – you're developing a unique linguistic fingerprint!\n\n")

    f.write("*Keep writing, keep growing, keep being awesome!* ✨\n\n")


def _write_repro_notes(f, analysis_params: Dict[str, Any], tokenizer_info: Dict[str, Any]) -> None:
    """Write reproducibility notes section"""
    f.write("## 9. Reproducibility Notes\n\n")

    f.write("### 🔄 Parameters Used\n\n")
    f.write("```json\n")
    f.write(json.dumps({
        'analysis': {k: v for k, v in analysis_params.items() if k in ['max_n', 'min_freq', 'collocation', 'seed']},
        'tokenizer': tokenizer_info.get('name', 'unknown')
    }, indent=2, ensure_ascii=False))
    f.write("\n```\n\n")

    f.write("### 📋 System Information\n\n")
    f.write(f"- **Generated**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    f.write(f"- **Tokenizer Backend**: {tokenizer_info.get('name', 'unknown')}\n")

    fallback_used = tokenizer_info.get('fallback_used', False)
    if fallback_used:
        f.write("- **Note**: pkuseg unavailable, jieba fallback used\n")

    f.write("\n---\n\n")
    f.write("*Report generated by Enhanced Linguistic Analysis Pipeline v2.0*\n")
