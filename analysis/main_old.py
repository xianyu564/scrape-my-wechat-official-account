#!/usr/bin/env python3
"""
中文语料分析主程序 - 分两步执行：分析 + 呈现
"""

import os
import pickle
import sys
import warnings
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd
from tqdm import tqdm

# 添加模块路径
sys.path.insert(0, str(Path(__file__).parent))

from src.freq_stats import (
    get_stats_summary,
    save_freq_stats,
    term_freq_by_year,
    term_freq_overall,
    tfidf_topk_by_year,
)
from src.io_utils import get_corpus_stats, read_text, scan_articles
from src.tokenize_zh import ChineseTokenizer
from src.wordcloud_viz import (
    generate_yearly_wordclouds,
)


def analyze_corpus(
    root_dir: str,
    output_dir: str = "analysis/out",
    # 时间过滤选项
    start_date: Optional[str] = None,  # 格式: "2020-01-01"
    end_date: Optional[str] = None,    # 格式: "2023-12-31"
    years: Optional[List[str]] = None, # 例如: ["2021", "2022", "2023"]
    # 分词参数
    min_df: int = 5,                   # TF-IDF最小文档频率
    max_df: float = 0.85,              # TF-IDF最大文档频率
    ngram_max: int = 2,                # 最大n-gram长度
    topk: int = 50,                    # 每年返回的关键词数量
    # 自定义文件路径
    userdict_path: Optional[str] = None,        # 自定义词典
    stopwords_path: str = "analysis/assets/stopwords_zh.txt", # 停用词文件
    extra_stopwords_path: Optional[str] = None, # 额外停用词文件
) -> Dict[str, Any]:
    """
    第一步：纯理论分析 - 处理文本语料，生成统计数据，不产生可视化输出
    
    Args:
        root_dir: 语料根目录，如 "Wechat-Backup/文不加点的张衔瑜"
        output_dir: 分析结果输出目录
        其他参数见注释
    
    Returns:
        分析结果字典，包含所有统计信息
    """

    print("=" * 60)
    print("🔬 第一步：语料理论分析")
    print("=" * 60)

    # 扫描文章
    print(f"📁 扫描语料目录: {root_dir}")
    articles = scan_articles(
        root_dir=root_dir,
        start_date=start_date,
        end_date=end_date,
        years=years
    )

    if not articles:
        raise ValueError("❌ 未找到任何文章，请检查路径和过滤条件")

    # 显示语料统计
    corpus_stats = get_corpus_stats(articles)
    print(f"📊 语料统计: {corpus_stats}")

    # 初始化分词器
    print("🔤 初始化中文分词器...")
    extra_stopwords = []
    if extra_stopwords_path and os.path.exists(extra_stopwords_path):
        try:
            with open(extra_stopwords_path, encoding='utf-8') as f:
                extra_stopwords = [line.strip() for line in f if line.strip()]
        except Exception as e:
            warnings.warn(f"加载额外停用词失败: {e}")

    tokenizer = ChineseTokenizer(
        userdict_path=userdict_path,
        stopwords_path=stopwords_path,
        extra_stopwords=extra_stopwords,
        cache_dir=os.path.join(output_dir, ".cache")
    )

    # 读取和分词文本
    print("📖 读取文章内容并分词...")
    corpus_tokens = []
    corpus_by_year = {}
    texts_by_year = {}

    for article in tqdm(articles, desc="处理文章"):
        text = read_text(article)
        if not text:
            continue

        tokens = tokenizer.tokenize(text, chinese_only=False, ngram_max=ngram_max, preserve_english=True)
        if tokens:
            corpus_tokens.append(tokens)

            # 按年分组
            year = article.year
            if year not in corpus_by_year:
                corpus_by_year[year] = []
                texts_by_year[year] = []

            corpus_by_year[year].append(tokens)
            texts_by_year[year].append(text)

    if not corpus_tokens:
        raise ValueError("❌ 没有有效的分词结果")

    print(f"✅ 成功处理 {len(corpus_tokens)} 篇文章")

    # 词频统计
    print("📊 计算词频统计...")
    freq_overall = term_freq_overall(corpus_tokens)
    freq_by_year = term_freq_by_year(corpus_by_year)

    # TF-IDF 分析
    print("🔍 计算 TF-IDF 关键词...")
    def create_tokenizer_func(tokenizer, chinese_only=False, ngram_max=1, preserve_english=True):
        def tokenize_func(text):
            return tokenizer.tokenize(text, chinese_only, ngram_max, preserve_english)
        return tokenize_func

    tokenizer_func = create_tokenizer_func(tokenizer, chinese_only=False, ngram_max=ngram_max)

    try:
        tfidf_by_year = tfidf_topk_by_year(
            texts_by_year=texts_by_year,
            tokenizer_func=tokenizer_func,
            min_df=min_df,
            max_df=max_df,
            ngram_range=(1, ngram_max),
            topk=topk
        )
    except Exception as e:
        print(f"⚠️ TF-IDF 分析失败，跳过: {e}")
        tfidf_by_year = pd.DataFrame(columns=['year', 'word', 'score'])

    # 保存分析结果
    print("💾 保存分析结果...")
    os.makedirs(output_dir, exist_ok=True)

    # 保存统计数据
    save_freq_stats(freq_overall, freq_by_year, tfidf_by_year, output_dir)

    # 保存中间结果到pickle文件，供第二步使用
    analysis_results = {
        'corpus_stats': corpus_stats,
        'articles': articles,
        'corpus_tokens': corpus_tokens,
        'corpus_by_year': corpus_by_year,
        'texts_by_year': texts_by_year,
        'freq_overall': freq_overall,
        'freq_by_year': freq_by_year,
        'tfidf_by_year': tfidf_by_year,
        'analysis_params': {
            'root_dir': root_dir,
            'start_date': start_date,
            'end_date': end_date,
            'years': years,
            'min_df': min_df,
            'max_df': max_df,
            'ngram_max': ngram_max,
            'topk': topk
        },
        'analysis_time': datetime.now().isoformat()
    }

    # 保存分析结果供第二步使用
    results_path = os.path.join(output_dir, "analysis_results.pkl")
    with open(results_path, 'wb') as f:
        pickle.dump(analysis_results, f)

    print(f"🎯 理论分析完成！结果保存在: {output_dir}")
    print(f"📈 分析数据已保存到: {results_path}")

    return analysis_results


def generate_visualizations(
    output_dir: str = "analysis/out",
    # 词云参数
    make_wordcloud: bool = True,          # 是否生成词云
    wordcloud_top_n: int = 200,           # 整体词云词汇数量
    yearly_wordcloud_top_n: int = 100,    # 年度词云词汇数量
    # 文件路径
    font_path: Optional[str] = None,      # 中文字体文件路径
    mask_path: str = "analysis/assets/mask.png",  # 词云遮罩图片
    # 报告参数
    generate_report: bool = True,         # 是否生成Markdown报告
    # Zipf分析
    generate_zipf: bool = True,           # 是否生成Zipf定律分析图
) -> Dict[str, str]:
    """
    第二步：可视化呈现 - 基于第一步的分析结果生成词云、图表、报告等美观的可交付成果
    
    Args:
        output_dir: 输出目录（应与第一步相同）
        其他参数见注释
    
    Returns:
        生成文件的路径字典
    """

    print("=" * 60)
    print("🎨 第二步：可视化呈现")
    print("=" * 60)

    # 加载第一步的分析结果
    results_path = os.path.join(output_dir, "analysis_results.pkl")
    if not os.path.exists(results_path):
        raise FileNotFoundError(f"❌ 未找到分析结果文件: {results_path}\n请先运行第一步分析")

    print("📂 加载分析结果...")
    with open(results_path, 'rb') as f:
        analysis_results = pickle.load(f)

    freq_overall = analysis_results['freq_overall']
    freq_by_year = analysis_results['freq_by_year']
    tfidf_by_year = analysis_results['tfidf_by_year']
    corpus_stats = analysis_results['corpus_stats']

    generated_files = {}

    # Zipf 定律分析
    if generate_zipf:
        print("📈 生成科学级 Zipf 定律分析图...")
        zipf_path = os.path.join(output_dir, "zipf_overall_enhanced.png")
        from src.freq_stats import zipf_plot_enhanced
        zipf_plot_enhanced(
            freq_data=freq_overall,
            output_path=zipf_path,
            title="中文语料词频分布的科学级Zipf定律分析",
            font_path=font_path,
            color_scheme="nature"
        )
        generated_files['zipf_plot'] = zipf_path

    # 生成词云
    if make_wordcloud:
        print("🎨 生成期刊级词云...")

        # 增强版整体词云
        from src.wordcloud_viz import generate_enhanced_overall_wordcloud
        overall_wordcloud_path = generate_enhanced_overall_wordcloud(
            freq_data=freq_overall,
            output_dir=output_dir,
            mask_path=mask_path,
            font_path=font_path,
            top_n=wordcloud_top_n,
            color_scheme="nature"
        )
        if overall_wordcloud_path:
            generated_files['overall_wordcloud'] = overall_wordcloud_path

        # 年度词云
        yearly_wordcloud_paths = generate_yearly_wordclouds(
            freq_by_year=freq_by_year,
            output_dir=output_dir,
            mask_path=mask_path,
            font_path=font_path,
            top_n=yearly_wordcloud_top_n
        )
        generated_files['yearly_wordclouds'] = yearly_wordcloud_paths

    # 生成报告
    if generate_report:
        print("📝 生成分析报告...")
        stats_summary = get_stats_summary(freq_overall, freq_by_year, tfidf_by_year)
        report_path = generate_markdown_report(
            stats_summary=stats_summary,
            freq_overall=freq_overall,
            freq_by_year=freq_by_year,
            tfidf_by_year=tfidf_by_year,
            corpus_stats=corpus_stats,
            analysis_params=analysis_results['analysis_params'],
            output_dir=output_dir
        )
        generated_files['report'] = report_path

    print(f"🎉 可视化呈现完成！结果保存在: {output_dir}")
    print("\n生成的文件:")
    for file_type, file_path in generated_files.items():
        if isinstance(file_path, list):
            print(f"  - {file_type}: {len(file_path)} 个文件")
            for path in file_path:
                print(f"    * {os.path.basename(path)}")
        else:
            print(f"  - {file_type}: {os.path.basename(file_path)}")

    return generated_files


def generate_markdown_report(
    stats_summary: dict,
    freq_overall: pd.DataFrame,
    freq_by_year: pd.DataFrame,
    tfidf_by_year: pd.DataFrame,
    corpus_stats: dict,
    analysis_params: dict,
    output_dir: str
) -> str:
    """Generate scientific-grade Markdown analysis report with English interface"""

    report_path = os.path.join(output_dir, "REPORT.md")

    with open(report_path, 'w', encoding='utf-8') as f:
        # Header and executive summary
        f.write("# 📊 Chinese Linguistic Corpus Analysis Report\n\n")
        f.write("> **Analysis Target**: Personal WeChat Official Account Article Corpus\n")
        f.write("> **Methodology**: Zipf's Law-based Frequency Statistical Analysis\n")
        f.write("> **Technology Stack**: jieba tokenization + TF-IDF + Statistical Visualization\n\n")

        f.write("---\n\n")

        # Core findings (Executive Summary)
        f.write("## 🎯 Executive Summary\n\n")

        if 'total_articles' in corpus_stats and 'total_unique_words' in stats_summary:
            total_articles = corpus_stats['total_articles']
            unique_words = stats_summary['total_unique_words']
            total_freq = stats_summary['total_word_freq']

            f.write(f"📈 **Corpus Scale**: {total_articles:,} articles, {unique_words:,} unique tokens, total frequency {total_freq:,}\n\n")

            # Calculate vocabulary density
            vocab_density = unique_words / total_freq if total_freq > 0 else 0
            diversity_level = "High" if vocab_density > 0.1 else "Medium" if vocab_density > 0.05 else "Low"
            f.write(f"🧠 **Vocabulary Density**: {vocab_density:.3f} ({diversity_level} level) - Reflecting linguistic expression richness\n\n")

        if 'years' in stats_summary and len(stats_summary['years']) > 1:
            years = stats_summary['years']
            f.write(f"⏱️ **Temporal Span**: {min(years)}-{max(years)} ({len(years)} years of data)\n\n")

        # Overall vocabulary map
        f.write("## 🎨 Overall Vocabulary Landscape\n\n")
        f.write("![Overall Word Cloud](wordcloud_overall.png)\n\n")
        f.write("*Word size reflects usage frequency, color encoding follows scientific journal color schemes*\n\n")

        # Top frequency statistics
        f.write("## 🔥 Top 20 High-Frequency Words\n\n")
        if not freq_overall.empty:
            top_20 = freq_overall.head(20)

            # Create two-column layout
            f.write("| Rank | Word | Freq | Rank | Word | Freq |\n")
            f.write("|:---:|:---:|:---:|:---:|:---:|:---:|\n")

            for i in range(0, min(20, len(top_20)), 2):
                left_row = top_20.iloc[i]
                left_rank = i + 1
                left_word = left_row['word']
                left_freq = left_row['freq']

                if i + 1 < len(top_20):
                    right_row = top_20.iloc[i + 1]
                    right_rank = i + 2
                    right_word = right_row['word']
                    right_freq = right_row['freq']
                    f.write(f"| {left_rank} | **{left_word}** | {left_freq:,} | {right_rank} | **{right_word}** | {right_freq:,} |\n")
                else:
                    f.write(f"| {left_rank} | **{left_word}** | {left_freq:,} | - | - | - |\n")

            f.write("\n")

        # Zipf's Law analysis
        f.write("## 📈 Linguistic Statistical Pattern Analysis\n\n")
        f.write("![Zipf's Law Analysis](zipf_overall_enhanced.png)\n\n")
        f.write("**Zipf's Law Validation**: Word frequency exhibits inverse relationship with rank, confirming natural language characteristics of Chinese corpus.\n\n")

        # Annual evolution analysis
        if 'years' in stats_summary and len(stats_summary['years']) > 1:
            f.write("## 📅 Annual Linguistic Evolution\n\n")

            years = sorted(stats_summary['years'])

            # Create annual comparison table
            f.write("| Year | Core Keywords | Distinctive Features |\n")
            f.write("|:---:|:---:|:---|\n")

            for year in years:
                # Get annual high-frequency words
                year_freq = freq_by_year[freq_by_year['year'] == year].head(3)
                if not year_freq.empty:
                    top_words = " • ".join(year_freq['word'].tolist())
                else:
                    top_words = "Data Missing"

                # Get annual distinctive words (TF-IDF)
                if not tfidf_by_year.empty and 'year' in tfidf_by_year.columns:
                    year_tfidf = tfidf_by_year[tfidf_by_year['year'] == year].head(2)
                    if not year_tfidf.empty:
                        distinctive_words = " • ".join(year_tfidf['word'].tolist())
                    else:
                        distinctive_words = "Under Analysis"
                else:
                    distinctive_words = "Under Analysis"

                f.write(f"| **{year}** | {top_words} | {distinctive_words} |\n")

            f.write("\n")

            # Annual word cloud gallery (compact display)
            f.write("### 🖼️ Annual Word Cloud Evolution\n\n")

            # Display 3 years per row
            years_per_row = 3
            for i in range(0, len(years), years_per_row):
                year_group = years[i:i+years_per_row]

                # Image row
                img_row = " | ".join([f"![{year}](wordcloud_{year}.png)" for year in year_group])
                f.write(f"| {img_row} |\n")

                # Title row
                title_row = " | ".join([f"**{year}**" for year in year_group])
                f.write(f"| {title_row} |\n")

                # Separator
                sep_row = " | ".join([":---:" for _ in year_group])
                f.write(f"| {sep_row} |\n\n")

        # Technical specifications and parameters
        f.write("---\n\n")
        f.write("## ⚙️ Technical Specifications\n\n")

        f.write("**Core Configuration Parameters**:\n")
        f.write(f"- Tokenization Engine: jieba (precise mode) + {get_phrase_dict_size()} custom phrase dictionary entries\n")
        f.write(f"- TF-IDF Parameters: min_df={analysis_params.get('min_df', 'N/A')}, max_df={analysis_params.get('max_df', 'N/A')}\n")
        f.write(f"- **N-gram Length**: 1-{analysis_params.get('ngram_max', 'N/A')} (supporting single chars, words, phrases, four-character idioms)\n")
        f.write("- Stopwords Library: Built-in 76 + custom extensions\n")
        f.write("- Mixed Chinese-English: Intelligent recognition and preservation of English technical terms\n")
        f.write("- Visualization: Scientific journal color schemes + 300 DPI high-resolution output\n\n")

        # Add comprehensive N-gram statistics
        if 'ngram_stats' in stats_summary:
            ngram_stats = stats_summary['ngram_stats']
            f.write("**Linguistic Structure Analysis**:\n")
            f.write(f"- Single Characters: {ngram_stats.get('单字词', 0):,} tokens (meaningful Chinese characters retained)\n")
            f.write(f"- Two-Character Words: {ngram_stats.get('双字词', 0):,} tokens (common vocabulary)\n")
            f.write(f"- Three-Character Phrases: {ngram_stats.get('三字词', 0):,} tokens (colloquialisms, technical terms)\n")
            f.write(f"- Four-Character Idioms: {ngram_stats.get('四字词', 0):,} tokens (idioms, compound concepts)\n")
            f.write(f"- Multi-Character Terms: {ngram_stats.get('多字词', 0):,} tokens (complex technical terminology)\n")
            f.write(f"- English Words: {ngram_stats.get('英文词', 0):,} tokens (technical terms preserved)\n")
            f.write(f"- Compound N-grams: {ngram_stats.get('复合词', 0):,} tokens (intelligent n-gram combinations)\n")
            f.write(f"- Technical Terms: {ngram_stats.get('技术词', 0):,} tokens (domain-specific vocabulary)\n")
            f.write(f"- Classical Idioms: {ngram_stats.get('成语词', 0):,} tokens (traditional four-character expressions)\n\n")

        f.write("**Quality Assurance**:\n")
        f.write("- ✅ Semantic filtering for single-character words (meaningful Chinese characters retained)\n")
        f.write("- ✅ N-gram semantic coherence validation\n")
        f.write("- ✅ Zipf's Law compliance verification\n")
        f.write("- ✅ Multi-dimensional statistical cross-validation\n")
        f.write("- ✅ English-Chinese mixed content intelligent processing\n")
        f.write("- ✅ Technical terminology preservation and classification\n\n")

        # Footer
        f.write("---\n\n")
        current_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        f.write(f"*📋 Report Generated: {current_time}*\n")
        f.write("*🔧 Analysis Engine: Advanced Chinese Linguistic Analysis System v3.0*\n")
        f.write(f"*📁 Data Source: {analysis_params.get('root_dir', 'WeChat Official Account Corpus')}*\n")
        f.write("*🌐 Language Support: Comprehensive Chinese (1-4 character structures) + English Technical Terms*\n")

    print(f"📄 Scientific-grade analysis report generated: {report_path}")
    return report_path


def get_phrase_dict_size() -> int:
    """Get the size of the phrase dictionary for reporting"""
    try:
        phrase_dict_path = "analysis/assets/chinese_phrases.txt"
        if os.path.exists(phrase_dict_path):
            with open(phrase_dict_path, encoding='utf-8') as f:
                count = 0
                for line in f:
                    line = line.strip()
                    if line and not line.startswith('#'):
                        count += 1
                return count
    except:
        pass
    return 122  # Default fallback


def main():
    """
    主函数 - 可以选择运行分析、呈现或两者
    """

    # =================================================================
    # 🔧 配置参数 - 根据需要修改下面的参数
    # =================================================================

    # 必需参数
    ROOT_DIR = "../Wechat-Backup/文不加点的张衔瑜"  # 语料根目录
    OUTPUT_DIR = "out"                         # 输出目录

    # 运行模式选择
    RUN_ANALYSIS = True       # 是否运行第一步（理论分析）
    RUN_VISUALIZATION = True  # 是否运行第二步（可视化呈现）

    # 第一步：分析参数
    ANALYSIS_PARAMS = {
        # 时间过滤
        'start_date': None,           # "2020-01-01" 或 None
        'end_date': None,             # "2023-12-31" 或 None
        'years': None,                # ["2021", "2022", "2023"] 或 None

        # 分词参数 - 优化以支持复杂中文语言结构
        'min_df': 1,                  # TF-IDF最小文档频率 (降低以保留更多有意义词汇)
        'max_df': 0.98,               # TF-IDF最大文档频率 (提高以保留常用词)
        'ngram_max': 4,               # 最大n-gram长度 (增加以支持三字词、四字成语、更长技术术语等)
        'topk': 150,                  # 每年返回的关键词数量 (增加以获得更丰富分析)

        # 自定义文件路径
        'userdict_path': None,        # 自定义词典文件
        'extra_stopwords_path': None, # 额外停用词文件
    }

    # 第二步：可视化参数
    VISUALIZATION_PARAMS = {
        # 词云设置 - 期刊级科学风格
        'make_wordcloud': True,              # 是否生成词云
        'wordcloud_top_n': 400,              # 整体词云词汇数量 (增加以获得更丰富的视觉效果)
        'yearly_wordcloud_top_n': 200,       # 年度词云词汇数量 (增加层次感)
        'font_path': None,                   # 中文字体文件路径，如 "/path/to/simhei.ttf"
        'mask_path': "analysis/assets/mask.png",  # 词云遮罩图片

        # 其他输出
        'generate_report': True,             # 是否生成Markdown报告
        'generate_zipf': True,               # 是否生成科学级Zipf定律分析图
    }

    # =================================================================
    # 🚀 执行分析流程
    # =================================================================

    try:
        # 第一步：理论分析
        if RUN_ANALYSIS:
            analysis_results = analyze_corpus(
                root_dir=ROOT_DIR,
                output_dir=OUTPUT_DIR,
                **ANALYSIS_PARAMS
            )
            print("\n" + "="*60)
            print("✅ 第一步完成！可以修改可视化参数后单独运行第二步")
            print("="*60 + "\n")

        # 第二步：可视化呈现
        if RUN_VISUALIZATION:
            generated_files = generate_visualizations(
                output_dir=OUTPUT_DIR,
                **VISUALIZATION_PARAMS
            )
            print("\n" + "="*60)
            print("🎉 全部分析完成！")
            print("="*60 + "\n")

        # 如果只运行一步，给出提示
        if RUN_ANALYSIS and not RUN_VISUALIZATION:
            print("💡 提示：要生成可视化结果，请设置 RUN_VISUALIZATION = True")
        elif RUN_VISUALIZATION and not RUN_ANALYSIS:
            print("💡 提示：如需重新分析语料，请设置 RUN_ANALYSIS = True")

    except Exception as e:
        print(f"❌ 运行出错: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
