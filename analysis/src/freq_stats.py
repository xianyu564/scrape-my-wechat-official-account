"""
词频统计与 TF-IDF 分析模块
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import font_manager
from sklearn.feature_extraction.text import TfidfVectorizer
from collections import Counter, defaultdict
from typing import List, Dict, Tuple, Optional
import warnings


def term_freq_overall(corpus_tokens: List[List[str]]) -> pd.DataFrame:
    """
    计算整体词频
    
    Args:
        corpus_tokens: 分词后的语料库，每个元素是一篇文章的词列表
    
    Returns:
        pd.DataFrame: 词频统计，列为 ['word', 'freq']
    """
    # 合并所有词
    all_words = []
    for tokens in corpus_tokens:
        all_words.extend(tokens)
    
    # 词频统计
    word_counts = Counter(all_words)
    
    # 转换为 DataFrame
    df = pd.DataFrame([
        {'word': word, 'freq': freq}
        for word, freq in word_counts.most_common()
    ])
    
    return df


def term_freq_by_year(corpus_by_year: Dict[str, List[List[str]]]) -> pd.DataFrame:
    """
    计算逐年词频
    
    Args:
        corpus_by_year: 按年份分组的语料，格式为 {year: [tokens_list]}
    
    Returns:
        pd.DataFrame: 逐年词频统计，列为 ['year', 'word', 'freq']
    """
    results = []
    
    for year, year_tokens in corpus_by_year.items():
        # 合并该年所有词
        year_words = []
        for tokens in year_tokens:
            year_words.extend(tokens)
        
        # 词频统计
        word_counts = Counter(year_words)
        
        # 添加到结果
        for word, freq in word_counts.items():
            results.append({
                'year': year,
                'word': word, 
                'freq': freq
            })
    
    df = pd.DataFrame(results)
    
    # 按年份和频率排序
    df = df.sort_values(['year', 'freq'], ascending=[True, False])
    
    return df


def tfidf_topk_by_year(texts_by_year: Dict[str, List[str]], 
                       tokenizer_func, 
                       min_df: int = 5,
                       max_df: float = 0.85,
                       ngram_range: Tuple[int, int] = (1, 2),
                       topk: int = 50) -> pd.DataFrame:
    """
    计算年度 TF-IDF 关键词
    
    Args:
        texts_by_year: 按年份分组的原始文本，格式为 {year: [text_list]}
        tokenizer_func: 分词函数
        min_df: 最小文档频率
        max_df: 最大文档频率
        ngram_range: n-gram 范围
        topk: 每年返回的关键词数量
    
    Returns:
        pd.DataFrame: TF-IDF 关键词，列为 ['year', 'word', 'score']
    """
    results = []
    
    for year, year_texts in texts_by_year.items():
        if not year_texts:
            continue
            
        try:
            # 合并年度文本
            year_corpus = ' '.join(year_texts)
            
            # 创建 TF-IDF 向量化器
            vectorizer = TfidfVectorizer(
                tokenizer=tokenizer_func,
                lowercase=False,
                min_df=max(1, min(min_df, len(year_texts))),  # 确保不超过文档数
                max_df=max_df,
                ngram_range=ngram_range,
                token_pattern=None  # 禁用默认正则，使用自定义分词
            )
            
            # 计算 TF-IDF
            tfidf_matrix = vectorizer.fit_transform([year_corpus])
            feature_names = vectorizer.get_feature_names_out()
            tfidf_scores = tfidf_matrix.toarray()[0]
            
            # 获取 top-k 关键词
            top_indices = np.argsort(tfidf_scores)[::-1][:topk]
            
            for idx in top_indices:
                if tfidf_scores[idx] > 0:  # 过滤零分词
                    results.append({
                        'year': year,
                        'word': feature_names[idx],
                        'score': tfidf_scores[idx]
                    })
        
        except Exception as e:
            warnings.warn(f"计算 {year} 年 TF-IDF 失败: {e}")
            continue
    
    df = pd.DataFrame(results)
    
    # 按年份和分数排序
    if not df.empty:
        df = df.sort_values(['year', 'score'], ascending=[True, False])
    
    return df


def zipf_plot(freq_data: pd.DataFrame, output_path: str, 
              title: str = "中文语料词频分布的Zipf定律分析",
              font_path: Optional[str] = None):
    """
    绘制高质量科学研究风格的 Zipf 定律分析图
    
    Args:
        freq_data: 词频数据，需包含 'freq' 列
        output_path: 输出图片路径
        title: 图表标题
        font_path: 中文字体路径
    """
    if freq_data.empty:
        warnings.warn("词频数据为空，无法绘制 Zipf 图")
        return
    
    # 设置科学研究风格
    plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
    
    # 设置字体
    if font_path and os.path.exists(font_path):
        font_prop = font_manager.FontProperties(fname=font_path)
        plt.rcParams['font.family'] = font_prop.get_name()
    else:
        plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'DejaVu Sans']
    
    plt.rcParams['axes.unicode_minus'] = False
    
    # 准备数据
    freqs = freq_data['freq'].values
    ranks = np.arange(1, len(freqs) + 1)
    
    # 过滤零频率
    valid_mask = freqs > 0
    freqs = freqs[valid_mask]
    ranks = ranks[valid_mask]
    
    if len(freqs) < 2:
        warnings.warn("有效词频数据不足，无法绘制 Zipf 图")
        return
    
    # 创建图表 - 使用黄金比例
    fig_width = 16
    fig_height = fig_width / 1.618  # 黄金比例
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(fig_width, fig_height))
    
    # 科学配色
    primary_color = '#2E86AB'
    secondary_color = '#A23B72' 
    accent_color = '#F18F01'
    
    # 左上图：双对数散点图
    ax1.loglog(ranks, freqs, 'o', color=primary_color, alpha=0.7, markersize=4)
    ax1.set_xlabel('词频排名', fontsize=12, fontweight='bold')
    ax1.set_ylabel('词频', fontsize=12, fontweight='bold')
    ax1.set_title('Zipf定律双对数分布', fontsize=14, fontweight='bold', color='#2C3E50')
    ax1.grid(True, alpha=0.3, linestyle='--')
    
    # 拟合直线
    log_ranks = np.log(ranks)
    log_freqs = np.log(freqs)
    
    # 线性回归
    coeff = np.polyfit(log_ranks, log_freqs, 1)
    slope, intercept = coeff
    
    # 绘制拟合线
    fitted_freqs = np.exp(intercept) * ranks ** slope
    ax1.loglog(ranks, fitted_freqs, '-', color=secondary_color, linewidth=3, 
               label=f'拟合线: y = {np.exp(intercept):.1f} × x^{slope:.3f}')
    ax1.legend(fontsize=11)
    
    # 计算 R²
    ss_res = np.sum((log_freqs - (slope * log_ranks + intercept)) ** 2)
    ss_tot = np.sum((log_freqs - np.mean(log_freqs)) ** 2)
    r_squared = 1 - (ss_res / ss_tot)
    
    # 右上图：残差分析
    residuals = log_freqs - (slope * log_ranks + intercept)
    ax2.scatter(log_ranks, residuals, alpha=0.6, s=20, color=accent_color)
    ax2.axhline(y=0, color=secondary_color, linestyle='--', alpha=0.8, linewidth=2)
    ax2.set_xlabel('log(排名)', fontsize=12, fontweight='bold')
    ax2.set_ylabel('残差', fontsize=12, fontweight='bold')
    ax2.set_title(f'残差分析 (R² = {r_squared:.4f})', fontsize=14, fontweight='bold', color='#2C3E50')
    ax2.grid(True, alpha=0.3, linestyle='--')
    
    # 左下图：累积分布
    cumulative_freq = np.cumsum(freqs) / np.sum(freqs)
    ax3.semilogx(ranks, cumulative_freq, color=primary_color, linewidth=2)
    ax3.set_xlabel('词频排名', fontsize=12, fontweight='bold')
    ax3.set_ylabel('累积频率比例', fontsize=12, fontweight='bold')
    ax3.set_title('词频累积分布', fontsize=14, fontweight='bold', color='#2C3E50')
    ax3.grid(True, alpha=0.3, linestyle='--')
    
    # 添加80-20标记线
    pareto_20_idx = int(len(ranks) * 0.2)
    pareto_freq_proportion = cumulative_freq[pareto_20_idx] if pareto_20_idx < len(cumulative_freq) else None
    if pareto_freq_proportion:
        ax3.axvline(x=ranks[pareto_20_idx], color=secondary_color, linestyle=':', alpha=0.8)
        ax3.axhline(y=pareto_freq_proportion, color=secondary_color, linestyle=':', alpha=0.8)
        ax3.text(ranks[pareto_20_idx], 0.1, f'前20%词汇\n占{pareto_freq_proportion:.1%}频次', 
                bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))
    
    # 右下图：词频分布直方图
    log_freq_bins = np.logspace(0, np.log10(freqs.max()), 30)
    ax4.hist(freqs, bins=log_freq_bins, alpha=0.7, color=accent_color, edgecolor='white')
    ax4.set_xscale('log')
    ax4.set_xlabel('词频', fontsize=12, fontweight='bold')
    ax4.set_ylabel('词汇数量', fontsize=12, fontweight='bold')
    ax4.set_title('词频分布直方图', fontsize=14, fontweight='bold', color='#2C3E50')
    ax4.grid(True, alpha=0.3, linestyle='--')
    
    # 主标题
    fig.suptitle(title, fontsize=18, fontweight='bold', y=0.95, color='#2C3E50')
    
    # 科学分析总结
    # 计算一些关键统计量
    median_freq = np.median(freqs)
    freq_variance = np.var(freqs)
    top_10_percent_freq = np.sum(freqs[:int(len(freqs)*0.1)]) / np.sum(freqs)
    
    zipf_analysis = (
        f"🔬 Zipf定律分析结果\n"
        f"• 拟合斜率 α = {-slope:.3f} (理论值 ≈ 1.0)\n"
        f"• 拟合优度 R² = {r_squared:.4f}\n"
        f"• 词汇多样性: {len(freqs):,} 个独特词汇\n"
        f"• 总词频: {freqs.sum():,} 次\n"
        f"• 频率中位数: {median_freq:.1f}\n"
        f"• 前10%词汇占总频次: {top_10_percent_freq:.1%}\n"
        f"• 语言分布符合度: {'优秀' if r_squared > 0.8 else '良好' if r_squared > 0.6 else '一般'}"
    )
    
    fig.text(0.02, 0.02, zipf_analysis, fontsize=11, family='monospace',
             bbox=dict(boxstyle="round,pad=0.5", facecolor="#F8F9FA", 
                      edgecolor='#BDC3C7', alpha=0.95))
    
    plt.tight_layout()
    plt.subplots_adjust(top=0.92, bottom=0.15)
    
    # 保存高质量图片
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    
    print(f"🎯 高质量Zipf分析图已保存: {output_path}")
    print(f"📊 分析结果: 斜率={-slope:.3f}, R²={r_squared:.4f}, 词汇数={len(freqs):,}")


def save_freq_stats(freq_overall: pd.DataFrame,
                   freq_by_year: pd.DataFrame,
                   tfidf_by_year: pd.DataFrame,
                   output_dir: str = "analysis/out"):
    """
    保存词频统计结果到 CSV 文件
    
    Args:
        freq_overall: 整体词频
        freq_by_year: 逐年词频
        tfidf_by_year: 年度 TF-IDF
        output_dir: 输出目录
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # 保存整体词频
    if not freq_overall.empty:
        overall_path = os.path.join(output_dir, "freq_overall.csv")
        freq_overall.to_csv(overall_path, index=False, encoding='utf-8')
        print(f"整体词频已保存: {overall_path}")
    
    # 保存逐年词频
    if not freq_by_year.empty:
        yearly_path = os.path.join(output_dir, "freq_by_year.csv")
        freq_by_year.to_csv(yearly_path, index=False, encoding='utf-8')
        print(f"逐年词频已保存: {yearly_path}")
    
    # 保存年度 TF-IDF
    if not tfidf_by_year.empty:
        tfidf_path = os.path.join(output_dir, "tfidf_topk_by_year.csv")
        tfidf_by_year.to_csv(tfidf_path, index=False, encoding='utf-8')
        print(f"年度 TF-IDF 已保存: {tfidf_path}")


def analyze_word_evolution(freq_by_year: pd.DataFrame, 
                          words: List[str]) -> pd.DataFrame:
    """
    分析特定词汇的年度演化趋势
    
    Args:
        freq_by_year: 逐年词频数据
        words: 要分析的词汇列表
    
    Returns:
        pd.DataFrame: 词汇演化数据，列为 ['year', 'word', 'freq', 'freq_norm']
    """
    if freq_by_year.empty:
        return pd.DataFrame()
    
    # 筛选目标词汇
    target_data = freq_by_year[freq_by_year['word'].isin(words)].copy()
    
    if target_data.empty:
        return pd.DataFrame()
    
    # 计算年度总词频用于归一化
    year_totals = freq_by_year.groupby('year')['freq'].sum().to_dict()
    
    # 添加归一化频率
    target_data['freq_norm'] = target_data.apply(
        lambda row: row['freq'] / year_totals.get(row['year'], 1), axis=1
    )
    
    return target_data.sort_values(['word', 'year'])


def get_stats_summary(freq_overall: pd.DataFrame,
                     freq_by_year: pd.DataFrame,
                     tfidf_by_year: pd.DataFrame) -> Dict:
    """
    获取统计摘要信息
    
    Args:
        freq_overall: 整体词频
        freq_by_year: 逐年词频
        tfidf_by_year: 年度 TF-IDF
    
    Returns:
        Dict: 统计摘要
    """
    summary = {}
    
    # 整体统计
    if not freq_overall.empty:
        summary['total_unique_words'] = len(freq_overall)
        summary['total_word_freq'] = freq_overall['freq'].sum()
        summary['top_words'] = freq_overall.head(10)['word'].tolist()
    
    # 年度统计
    if not freq_by_year.empty:
        years = sorted(freq_by_year['year'].unique())
        summary['years'] = years
        summary['words_by_year'] = freq_by_year.groupby('year')['word'].nunique().to_dict()
        summary['freq_by_year'] = freq_by_year.groupby('year')['freq'].sum().to_dict()
    
    # TF-IDF 统计
    if not tfidf_by_year.empty:
        summary['tfidf_years'] = sorted(tfidf_by_year['year'].unique())
        summary['top_tfidf_words'] = tfidf_by_year.groupby('year').head(5).groupby('year')['word'].apply(list).to_dict()
    
    return summary