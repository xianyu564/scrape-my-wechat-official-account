#!/usr/bin/env python3
"""
自定义词云生成工具 - Custom Word Cloud Generator
支持按年份、时间段或完整数据集生成词云
Supports generating word clouds by year, time period, or complete dataset
"""

import argparse
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional

# Add pipeline to path
sys.path.insert(0, str(Path(__file__).parent / "pipeline"))

from corpus_io import load_corpus, read_text, split_by_year
from tokenizer import MixedLanguageTokenizer
from stats import calculate_frequencies, calculate_frequencies_by_year
from viz import create_wordcloud, setup_chinese_font


def generate_wordcloud_for_years(
    corpus_root: str,
    output_dir: str,
    years: Optional[List[str]] = None,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    max_words: int = 200,
    font_path: Optional[str] = None,
    color_scheme: str = "nature"
) -> None:
    """
    为指定年份或时间段生成词云
    Generate word clouds for specified years or time period
    
    Args:
        corpus_root: 语料库根目录 / Corpus root directory
        output_dir: 输出目录 / Output directory  
        years: 年份列表 / List of years
        start_date: 开始日期 (YYYY-MM-DD) / Start date
        end_date: 结束日期 (YYYY-MM-DD) / End date
        max_words: 最大词数 / Maximum words
        font_path: 中文字体路径 / Chinese font path
        color_scheme: 颜色方案 / Color scheme
    """
    
    print("🚀 开始生成自定义词云 / Starting custom word cloud generation")
    print(f"📁 语料库路径 / Corpus path: {corpus_root}")
    print(f"📅 年份筛选 / Year filter: {years if years else '所有年份 / All years'}")
    print(f"📆 日期范围 / Date range: {start_date} ~ {end_date}")
    
    # 创建输出目录 / Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # 加载语料库 / Load corpus
    print("📚 正在加载语料库... / Loading corpus...")
    articles = load_corpus(
        root_dir=corpus_root,
        start_date=start_date,
        end_date=end_date,
        years=years
    )
    
    if not articles:
        print("❌ 未找到文章 / No articles found")
        return
    
    print(f"✅ 加载了 {len(articles)} 篇文章 / Loaded {len(articles)} articles")
    
    # 初始化分词器 / Initialize tokenizer
    print("🔤 初始化分词器... / Initializing tokenizer...")
    tokenizer = MixedLanguageTokenizer()
    
    # 按年份分组 / Split by year
    articles_by_year = split_by_year(articles)
    
    # 处理所有文章数据 / Process all article data
    all_tokens = []
    tokens_by_year = {}
    
    for year, year_articles in articles_by_year.items():
        year_tokens = []
        print(f"📝 正在处理 {year} 年的 {len(year_articles)} 篇文章... / Processing {len(year_articles)} articles from {year}...")
        
        for article in year_articles:
            text = read_text(article)
            if text:
                tokens = tokenizer.tokenize(text)
                if tokens:
                    all_tokens.append(tokens)
                    year_tokens.append(tokens)
        
        if year_tokens:
            tokens_by_year[year] = year_tokens
            print(f"✅ {year} 年: {len(year_tokens)} 篇文章已分词 / {year}: {len(year_tokens)} articles tokenized")
    
    # 计算词频 / Calculate frequencies
    print("📊 计算词频统计... / Calculating frequency statistics...")
    freq_overall = calculate_frequencies(all_tokens)
    freq_by_year = calculate_frequencies_by_year(tokens_by_year)
    
    # 设置字体 / Setup font
    font_path = setup_chinese_font(font_path)
    
    # 生成整体词云 / Generate overall word cloud
    if len(all_tokens) > 0:
        overall_cloud_path = os.path.join(output_dir, "cloud_complete.png")
        print("🎨 生成完整数据词云... / Generating complete dataset word cloud...")
        
        date_range = ""
        if start_date and end_date:
            date_range = f" ({start_date} ~ {end_date})"
        elif years:
            date_range = f" ({', '.join(sorted(years))})"
        
        create_wordcloud(
            frequencies=freq_overall,
            output_path=overall_cloud_path,
            title=f"完整数据词云 / Complete Word Cloud{date_range}",
            font_path=font_path,
            max_words=max_words,
            color_scheme=color_scheme
        )
        print(f"✅ 完整词云已保存: {overall_cloud_path}")
    
    # 生成各年份词云 / Generate yearly word clouds
    if len(freq_by_year) > 1:  # 如果有多个年份 / If multiple years
        print("🎨 生成年度词云... / Generating yearly word clouds...")
        for year, freq in freq_by_year.items():
            if freq:  # 确保有数据 / Ensure there's data
                # 检查是否存在原始的词云文件，避免覆盖
                # Check if original word cloud exists to avoid overwriting
                original_cloud_path = os.path.join(output_dir, f"cloud_{year}.png")
                if os.path.exists(original_cloud_path):
                    print(f"⚠️  跳过 {year} 年: 原始词云已存在 / Skipping {year}: original word cloud exists")
                    continue
                    
                yearly_cloud_path = os.path.join(output_dir, f"cloud_{year}.png")
                create_wordcloud(
                    frequencies=freq,
                    output_path=yearly_cloud_path,
                    title=f"{year} 年词云 / {year} Word Cloud",
                    font_path=font_path,
                    max_words=max_words,
                    color_scheme=color_scheme
                )
                print(f"✅ {year} 年词云已保存: {yearly_cloud_path}")
    
    print("🎉 词云生成完成! / Word cloud generation completed!")
    print(f"📁 输出目录: {output_dir}")


def main():
    """主函数 / Main function"""
    parser = argparse.ArgumentParser(
        description="自定义词云生成工具 / Custom Word Cloud Generator",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
使用示例 / Usage Examples:

# 生成所有年份的词云 / Generate word clouds for all years:
python generate_wordclouds.py

# 生成特定年份的词云 / Generate word clouds for specific years:
python generate_wordclouds.py --years 2020,2021,2022,2023,2024,2025

# 生成指定时间段的词云 / Generate word clouds for date range:
python generate_wordclouds.py --start-date 2020-01-01 --end-date 2022-12-31

# 自定义输出目录和参数 / Custom output and parameters:
python generate_wordclouds.py --output custom_clouds --max-words 300 --color-scheme science
        """
    )
    
    # 基本参数 / Basic parameters
    parser.add_argument('--corpus', type=str, 
                       default="../Wechat-Backup/文不加点的张衔瑜",
                       help='语料库根目录 / Corpus root directory')
    parser.add_argument('--output', type=str,
                       default="out",
                       help='输出目录 / Output directory')
    
    # 时间筛选 / Time filtering
    parser.add_argument('--years', type=str,
                       help='年份列表(逗号分隔) / Year list (comma-separated), e.g., 2020,2021,2022')
    parser.add_argument('--start-date', type=str,
                       help='开始日期 / Start date (YYYY-MM-DD)')
    parser.add_argument('--end-date', type=str,
                       help='结束日期 / End date (YYYY-MM-DD)')
    
    # 词云参数 / Word cloud parameters
    parser.add_argument('--max-words', type=int, default=200,
                       help='最大词数 / Maximum words (default: 200)')
    parser.add_argument('--font-path', type=str,
                       help='中文字体路径 / Chinese font path')
    parser.add_argument('--color-scheme', choices=['nature', 'science', 'calm', 'muted', 'solar'],
                       default='nature',
                       help='颜色方案 / Color scheme (default: nature)')
    
    args = parser.parse_args()
    
    # 解析年份参数 / Parse years parameter
    years = None
    if args.years:
        years = [year.strip() for year in args.years.split(',')]
    
    # 生成词云 / Generate word clouds
    generate_wordcloud_for_years(
        corpus_root=args.corpus,
        output_dir=args.output,
        years=years,
        start_date=args.start_date,
        end_date=args.end_date,
        max_words=args.max_words,
        font_path=args.font_path,
        color_scheme=args.color_scheme
    )


if __name__ == "__main__":
    main()