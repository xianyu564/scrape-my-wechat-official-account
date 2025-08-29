#!/usr/bin/env python3
"""
中文语料分析命令行工具
"""

import os
import sys
import argparse
import warnings
from pathlib import Path
from typing import List, Optional
import pandas as pd
from tqdm import tqdm

# 添加模块路径
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.io_utils import scan_articles, read_text, get_corpus_stats
from src.tokenize_zh import ChineseTokenizer
from src.freq_stats import (
    term_freq_overall, term_freq_by_year, tfidf_topk_by_year,
    zipf_plot, save_freq_stats, get_stats_summary
)
from src.wordcloud_viz import (
    generate_overall_wordcloud, generate_yearly_wordclouds,
    create_wordcloud_comparison
)


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(
        description="中文语料的词频分析、TF-IDF 关键词与词云可视化",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例用法:
  # 基础分析
  python -m analysis.src.cli --root Wechat-Backup/文不加点的张衔瑜
  
  # 带时间过滤
  python -m analysis.src.cli --root Wechat-Backup/文不加点的张衔瑜 --start 2020-01-01 --end 2023-12-31
  
  # 指定年份和参数
  python -m analysis.src.cli --root Wechat-Backup/文不加点的张衔瑜 --years 2021,2022,2023 --topk 100
        """
    )
    
    # 必需参数
    parser.add_argument('--root', required=True,
                       help='语料根目录，如 Wechat-Backup/文不加点的张衔瑜')
    
    # 时间过滤
    parser.add_argument('--start', 
                       help='开始日期，格式 YYYY-MM-DD')
    parser.add_argument('--end',
                       help='结束日期，格式 YYYY-MM-DD') 
    parser.add_argument('--years',
                       help='年份白名单，逗号分隔，如 2017,2018,2019')
    
    # 分词参数
    parser.add_argument('--min-df', type=int, default=5,
                       help='TF-IDF 最小文档频率 (default: 5)')
    parser.add_argument('--max-df', type=float, default=0.85,
                       help='TF-IDF 最大文档频率 (default: 0.85)')
    parser.add_argument('--ngram-max', type=int, default=2,
                       help='最大 n-gram 长度 (default: 2)')
    parser.add_argument('--topk', type=int, default=50,
                       help='每年返回的关键词数量 (default: 50)')
    
    # 可选文件路径
    parser.add_argument('--font-path',
                       help='中文字体文件路径')
    parser.add_argument('--userdict',
                       help='自定义词典文件路径')
    parser.add_argument('--extra-stopwords',
                       help='额外停用词文件路径')
    
    # 输出选项
    parser.add_argument('--make-wordcloud', type=int, default=1,
                       help='是否生成词云 (1=是, 0=否, default: 1)')
    parser.add_argument('--report', type=int, default=1,
                       help='是否生成报告 (1=是, 0=否, default: 1)')
    parser.add_argument('--output-dir', default='analysis/out',
                       help='输出目录 (default: analysis/out)')
    
    return parser.parse_args()


def load_extra_stopwords(file_path: str) -> List[str]:
    """加载额外停用词"""
    if not os.path.exists(file_path):
        return []
    
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return [line.strip() for line in f if line.strip()]
    except Exception as e:
        warnings.warn(f"加载额外停用词失败: {e}")
        return []


def create_tokenizer_func(tokenizer, chinese_only=True, ngram_max=1):
    """创建分词函数供 TF-IDF 使用"""
    def tokenize_func(text):
        return tokenizer.tokenize(text, chinese_only, ngram_max)
    return tokenize_func


def generate_report(stats_summary: dict, 
                   freq_overall: pd.DataFrame,
                   freq_by_year: pd.DataFrame, 
                   tfidf_by_year: pd.DataFrame,
                   output_dir: str) -> str:
    """生成 Markdown 分析报告"""
    
    report_path = os.path.join(output_dir, "REPORT.md")
    
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("# 中文语料分析报告\n\n")
        
        # 概览统计
        f.write("## 📊 语料概览\n\n")
        if 'total_unique_words' in stats_summary:
            f.write(f"- **总词汇数**: {stats_summary['total_unique_words']:,}\n")
            f.write(f"- **总词频**: {stats_summary['total_word_freq']:,}\n")
        
        if 'years' in stats_summary:
            f.write(f"- **年份范围**: {min(stats_summary['years'])} - {max(stats_summary['years'])}\n")
            f.write(f"- **覆盖年数**: {len(stats_summary['years'])} 年\n\n")
        
        # 整体词频 Top 50
        f.write("## 🔥 整体高频词汇 (Top 50)\n\n")
        if not freq_overall.empty:
            top_50 = freq_overall.head(50)
            f.write("| 排名 | 词汇 | 频次 |\n")
            f.write("|------|------|------|\n")
            for i, row in top_50.iterrows():
                f.write(f"| {i+1} | {row['word']} | {row['freq']:,} |\n")
            f.write("\n")
        
        # 整体词云
        f.write("## 🎨 整体词云\n\n")
        f.write("![整体词云](wordcloud_overall.png)\n\n")
        
        # 年度分析
        if 'years' in stats_summary:
            f.write("## 📅 年度词频分析\n\n")
            
            for year in sorted(stats_summary['years']):
                f.write(f"### {year} 年\n\n")
                
                # 年度 Top 20 词频
                year_freq = freq_by_year[freq_by_year['year'] == year].head(20)
                if not year_freq.empty:
                    f.write("**高频词汇 (Top 20)**:\n\n")
                    f.write("| 排名 | 词汇 | 频次 |\n")
                    f.write("|------|------|------|\n")
                    for i, row in year_freq.iterrows():
                        rank = list(year_freq.index).index(i) + 1
                        f.write(f"| {rank} | {row['word']} | {row['freq']:,} |\n")
                    f.write("\n")
                
                # 年度 TF-IDF Top 20  
                if not tfidf_by_year.empty and 'year' in tfidf_by_year.columns:
                    year_tfidf = tfidf_by_year[tfidf_by_year['year'] == year].head(20)
                    if not year_tfidf.empty:
                        f.write("**TF-IDF 关键词 (Top 20)**:\n\n")
                        f.write("| 排名 | 词汇 | TF-IDF 分数 |\n")
                        f.write("|------|------|-------------|\n")
                        for i, row in year_tfidf.iterrows():
                            rank = list(year_tfidf.index).index(i) + 1
                            f.write(f"| {rank} | {row['word']} | {row['score']:.4f} |\n")
                        f.write("\n")
                
                # 年度词云
                f.write(f"**{year} 年词云**:\n\n")
                f.write(f"![{year}年词云](wordcloud_{year}.png)\n\n")
        
        # Zipf 定律分析
        f.write("## 📈 Zipf 定律分析\n\n")
        f.write("根据 Zipf 定律，词频与其排名呈反比关系（f ∝ 1/r^α）。")
        f.write("理想情况下，拟合斜率应接近 -1。\n\n")
        f.write("![Zipf定律分析](zipf_overall.png)\n\n")
        f.write("**解读**: 语言风格体现了典型的长尾分布特征，")
        f.write("少数高频词承载主要语言信息，大量低频词丰富表达的细节和个性。\n\n")
        
        # 年度口号生成
        if 'top_tfidf_words' in stats_summary and stats_summary['top_tfidf_words']:
            f.write("## 🎯 年度关键词口号\n\n")
            for year, words in stats_summary['top_tfidf_words'].items():
                if words:
                    top_3 = words[:3]
                    slogan = f"在{year}年，我专注于{top_3[0]}，深入探索{top_3[1] if len(top_3) > 1 else '新领域'}，持续思考{top_3[2] if len(top_3) > 2 else '人生哲理'}。"
                    f.write(f"**{year}**: {slogan}\n\n")
        
        # 生成时间
        from datetime import datetime
        f.write(f"\n---\n\n*报告生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*\n")
    
    print(f"分析报告已生成: {report_path}")
    return report_path


def main():
    """主函数"""
    args = parse_args()
    
    print("🚀 开始中文语料分析...")
    
    # 解析年份列表
    years = None
    if args.years:
        years = [year.strip() for year in args.years.split(',')]
    
    # 扫描文章
    print(f"📁 扫描语料目录: {args.root}")
    articles = scan_articles(
        root_dir=args.root,
        start_date=args.start,
        end_date=args.end,
        years=years
    )
    
    if not articles:
        print("❌ 未找到任何文章，请检查路径和过滤条件")
        return
    
    # 显示语料统计
    corpus_stats = get_corpus_stats(articles)
    print(f"📊 语料统计: {corpus_stats}")
    
    # 初始化分词器
    print("🔤 初始化中文分词器...")
    extra_stopwords = []
    if args.extra_stopwords:
        extra_stopwords = load_extra_stopwords(args.extra_stopwords)
    
    tokenizer = ChineseTokenizer(
        userdict_path=args.userdict,
        stopwords_path="analysis/assets/stopwords_zh.txt",
        extra_stopwords=extra_stopwords
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
        
        tokens = tokenizer.tokenize(text, chinese_only=True, ngram_max=args.ngram_max)
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
        print("❌ 没有有效的分词结果")
        return
    
    print(f"✅ 成功处理 {len(corpus_tokens)} 篇文章")
    
    # 词频统计
    print("📊 计算词频统计...")
    freq_overall = term_freq_overall(corpus_tokens)
    freq_by_year = term_freq_by_year(corpus_by_year)
    
    # TF-IDF 分析
    print("🔍 计算 TF-IDF 关键词...")
    tokenizer_func = create_tokenizer_func(tokenizer, chinese_only=True, ngram_max=args.ngram_max)
    
    # 简化的 TF-IDF 分析 - 如果失败就创建空的 DataFrame
    try:
        tfidf_by_year = tfidf_topk_by_year(
            texts_by_year=texts_by_year,
            tokenizer_func=tokenizer_func,
            min_df=args.min_df,
            max_df=args.max_df,
            ngram_range=(1, args.ngram_max),
            topk=args.topk
        )
    except Exception as e:
        print(f"⚠️ TF-IDF 分析失败，跳过: {e}")
        tfidf_by_year = pd.DataFrame(columns=['year', 'word', 'score'])
    
    # 保存统计结果
    print("💾 保存统计结果...")
    os.makedirs(args.output_dir, exist_ok=True)
    save_freq_stats(freq_overall, freq_by_year, tfidf_by_year, args.output_dir)
    
    # Zipf 定律分析
    print("📈 生成 Zipf 定律分析图...")
    zipf_plot(
        freq_data=freq_overall,
        output_path=os.path.join(args.output_dir, "zipf_overall.png"),
        font_path=args.font_path
    )
    
    # 生成词云
    if args.make_wordcloud:
        print("🎨 生成词云...")
        
        # 整体词云
        generate_overall_wordcloud(
            freq_data=freq_overall,
            output_dir=args.output_dir,
            mask_path="analysis/assets/mask.png",
            font_path=args.font_path,
            top_n=200
        )
        
        # 年度词云
        generate_yearly_wordclouds(
            freq_by_year=freq_by_year,
            output_dir=args.output_dir,
            mask_path="analysis/assets/mask.png", 
            font_path=args.font_path,
            top_n=100
        )
    
    # 生成报告
    if args.report:
        print("📝 生成分析报告...")
        stats_summary = get_stats_summary(freq_overall, freq_by_year, tfidf_by_year)
        generate_report(
            stats_summary=stats_summary,
            freq_overall=freq_overall,
            freq_by_year=freq_by_year,
            tfidf_by_year=tfidf_by_year,
            output_dir=args.output_dir
        )
    
    print(f"🎉 分析完成！结果保存在: {args.output_dir}")
    print("\n生成的文件:")
    for file_name in os.listdir(args.output_dir):
        print(f"  - {file_name}")


if __name__ == "__main__":
    main()