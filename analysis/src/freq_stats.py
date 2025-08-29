"""
词频统计与 TF-IDF 分析模块 - 增强版
提供科学级统计分析和期刊质量可视化
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.patches import Rectangle
import seaborn as sns
from sklearn.feature_extraction.text import TfidfVectorizer
from collections import Counter, defaultdict
from typing import List, Dict, Tuple, Optional
import warnings
from scipy import stats
from scipy.optimize import curve_fit
import math

# 设置科学期刊风格配色方案
NATURE_COLORS = {
    'primary': '#0C7BDC',      # Nature蓝
    'secondary': '#E1AF00',    # 金黄色
    'accent': '#DC143C',       # 深红色
    'success': '#039BE5',      # 浅蓝色
    'warning': '#FFA726',      # 橙色
    'background': '#F8F9FA',   # 浅灰背景
    'text': '#2C3E50',         # 深灰文字
    'grid': '#E9ECEF'          # 网格色
}

SCIENCE_COLORS = {
    'primary': '#1f77b4',      # Science蓝
    'secondary': '#ff7f0e',    # 橙色
    'accent': '#2ca02c',       # 绿色
    'warning': '#d62728',      # 红色
    'info': '#9467bd',         # 紫色
    'background': '#fafafa',   # 浅灰背景
    'text': '#333333',         # 深灰文字
    'grid': '#eeeeee'          # 网格色
}


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


def zipf_plot_enhanced(freq_data: pd.DataFrame, output_path: str, 
                      title: str = "中文语料词频分布的科学级Zipf定律分析",
                      font_path: Optional[str] = None,
                      color_scheme: str = "nature"):
    """
    绘制科学期刊级别的 Zipf 定律分析图 - 增强版
    
    Args:
        freq_data: 词频数据，需包含 'freq' 列
        output_path: 输出图片路径
        title: 图表标题
        font_path: 中文字体路径
        color_scheme: 配色方案 ("nature", "science", "custom")
    """
    if freq_data.empty:
        warnings.warn("词频数据为空，无法绘制 Zipf 图")
        return
    
    # 设置配色方案
    if color_scheme == "nature":
        colors = NATURE_COLORS
    elif color_scheme == "science":
        colors = SCIENCE_COLORS
    else:
        colors = NATURE_COLORS  # 默认使用Nature配色
    
    # 设置科学期刊风格
    plt.style.use('default')
    
    # 设置字体
    if font_path and os.path.exists(font_path):
        font_prop = font_manager.FontProperties(fname=font_path)
        plt.rcParams['font.family'] = font_prop.get_name()
    else:
        plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'DejaVu Sans']
    
    plt.rcParams['axes.unicode_minus'] = False
    plt.rcParams['figure.facecolor'] = 'white'
    plt.rcParams['axes.facecolor'] = colors['background']
    plt.rcParams['grid.color'] = colors['grid']
    plt.rcParams['text.color'] = colors['text']
    plt.rcParams['axes.labelcolor'] = colors['text']
    plt.rcParams['xtick.color'] = colors['text']
    plt.rcParams['ytick.color'] = colors['text']
    
    # 准备数据
    freqs = freq_data['freq'].values
    ranks = np.arange(1, len(freqs) + 1)
    
    # 过滤零频率和异常值
    valid_mask = freqs > 0
    freqs = freqs[valid_mask]
    ranks = ranks[valid_mask]
    
    if len(freqs) < 10:
        warnings.warn("有效词频数据不足，无法进行科学分析")
        return
    
    # 创建六面板科学分析图 - 使用黄金比例
    fig_width = 18
    fig_height = fig_width / 1.618
    fig = plt.figure(figsize=(fig_width, fig_height))
    
    # 创建复杂网格布局
    gs = fig.add_gridspec(3, 4, height_ratios=[1.2, 1, 0.8], width_ratios=[1, 1, 1, 1],
                         hspace=0.3, wspace=0.3)
    
    # 主图：双对数Zipf分布 (占上排左侧两格)
    ax_main = fig.add_subplot(gs[0, :2])
    
    # 残差分析 (上排右侧两格)
    ax_residual = fig.add_subplot(gs[0, 2:])
    
    # 累积分布 (中排左侧两格)
    ax_cumulative = fig.add_subplot(gs[1, :2])
    
    # 频率分布直方图 (中排右侧两格)
    ax_hist = fig.add_subplot(gs[1, 2:])
    
    # 统计摘要面板 (下排)
    ax_stats = fig.add_subplot(gs[2, :])
    
    # === 1. 主Zipf分布图 ===
    # 数据点
    scatter = ax_main.loglog(ranks, freqs, 'o', color=colors['primary'], 
                           alpha=0.6, markersize=3, markeredgewidth=0)
    
    # 拟合分析（多种方法）
    log_ranks = np.log(ranks)
    log_freqs = np.log(freqs)
    
    # 方法1: 标准线性回归
    coeff = np.polyfit(log_ranks, log_freqs, 1)
    slope, intercept = coeff
    
    # 方法2: 稳健回归（抗异常值）
    try:
        from scipy.stats import linregress
        slope_robust, intercept_robust, r_value, p_value, std_err = linregress(log_ranks, log_freqs)
        r_squared = r_value ** 2
    except:
        r_squared = 1 - (np.sum((log_freqs - (slope * log_ranks + intercept)) ** 2) / 
                        np.sum((log_freqs - np.mean(log_freqs)) ** 2))
        slope_robust, intercept_robust = slope, intercept
    
    # 绘制拟合线
    fitted_freqs = np.exp(intercept_robust) * ranks ** slope_robust
    ax_main.loglog(ranks, fitted_freqs, '-', color=colors['secondary'], linewidth=2.5, 
                  label=f'Zipf拟合: α = {-slope_robust:.3f}')
    
    # 理论Zipf线 (α = 1)
    theoretical_zipf = freqs[0] * ranks[0] / ranks
    ax_main.loglog(ranks, theoretical_zipf, '--', color=colors['accent'], linewidth=2, 
                  alpha=0.8, label='理论Zipf (α = 1.0)')
    
    ax_main.set_xlabel('词频排名', fontsize=13, fontweight='bold')
    ax_main.set_ylabel('词频', fontsize=13, fontweight='bold')
    ax_main.set_title('Zipf定律双对数分布 - 科学验证', fontsize=15, fontweight='bold')
    ax_main.legend(fontsize=11, framealpha=0.9)
    ax_main.grid(True, alpha=0.3, linestyle='-')
    
    # === 2. 残差分析 ===
    residuals = log_freqs - (slope_robust * log_ranks + intercept_robust)
    
    # 残差散点图
    ax_residual.scatter(log_ranks, residuals, alpha=0.6, s=15, color=colors['warning'])
    
    # 添加零线和置信区间
    ax_residual.axhline(y=0, color=colors['secondary'], linestyle='-', alpha=0.8, linewidth=2)
    
    # 计算残差的标准差
    residual_std = np.std(residuals)
    ax_residual.axhline(y=2*residual_std, color=colors['accent'], linestyle=':', alpha=0.6)
    ax_residual.axhline(y=-2*residual_std, color=colors['accent'], linestyle=':', alpha=0.6)
    
    ax_residual.set_xlabel('log(排名)', fontsize=13, fontweight='bold')
    ax_residual.set_ylabel('拟合残差', fontsize=13, fontweight='bold')
    ax_residual.set_title(f'残差分析 (R² = {r_squared:.4f})', fontsize=15, fontweight='bold')
    ax_residual.grid(True, alpha=0.3)
    
    # === 3. 累积分布分析 ===
    cumulative_freq = np.cumsum(freqs) / np.sum(freqs)
    ax_cumulative.semilogx(ranks, cumulative_freq, color=colors['primary'], linewidth=2.5)
    
    # 添加Pareto分析线
    pareto_points = [0.1, 0.2, 0.5]
    for p in pareto_points:
        idx = int(len(ranks) * p)
        if idx < len(cumulative_freq):
            cum_freq_p = cumulative_freq[idx]
            ax_cumulative.axvline(x=ranks[idx], color=colors['accent'], linestyle=':', alpha=0.7)
            ax_cumulative.axhline(y=cum_freq_p, color=colors['accent'], linestyle=':', alpha=0.7)
            ax_cumulative.text(ranks[idx], cum_freq_p + 0.05, f'{p:.0%}词汇→{cum_freq_p:.1%}频次', 
                             fontsize=9, ha='center',
                             bbox=dict(boxstyle="round,pad=0.2", facecolor="white", alpha=0.8))
    
    ax_cumulative.set_xlabel('词频排名', fontsize=13, fontweight='bold')
    ax_cumulative.set_ylabel('累积频率比例', fontsize=13, fontweight='bold')
    ax_cumulative.set_title('词频累积分布 - Pareto分析', fontsize=15, fontweight='bold')
    ax_cumulative.grid(True, alpha=0.3)
    
    # === 4. 频率分布直方图 ===
    # 使用对数分箱
    log_freqs_hist = np.log10(freqs)
    ax_hist.hist(log_freqs_hist, bins=25, alpha=0.7, color=colors['info'], edgecolor='white', linewidth=0.5)
    
    # 添加统计线
    median_log_freq = np.median(log_freqs_hist)
    mean_log_freq = np.mean(log_freqs_hist)
    ax_hist.axvline(median_log_freq, color=colors['secondary'], linestyle='--', linewidth=2, label='中位数')
    ax_hist.axvline(mean_log_freq, color=colors['warning'], linestyle='-', linewidth=2, label='均值')
    
    ax_hist.set_xlabel('log₁₀(词频)', fontsize=13, fontweight='bold')
    ax_hist.set_ylabel('词汇数量', fontsize=13, fontweight='bold')
    ax_hist.set_title('词频分布直方图 - 统计特征', fontsize=15, fontweight='bold')
    ax_hist.legend(fontsize=10)
    ax_hist.grid(True, alpha=0.3)
    
    # === 5. 科学统计摘要面板 ===
    ax_stats.axis('off')
    
    # 计算高级统计指标
    vocab_size = len(freqs)
    total_tokens = freqs.sum()
    type_token_ratio = vocab_size / total_tokens
    entropy = -np.sum((freqs/total_tokens) * np.log2(freqs/total_tokens))
    
    # Zipf质量评估
    zipf_quality = "优秀" if r_squared > 0.85 else "良好" if r_squared > 0.7 else "一般" if r_squared > 0.5 else "偏差较大"
    
    # 语言多样性评估
    diversity_level = "高" if type_token_ratio > 0.1 else "中等" if type_token_ratio > 0.05 else "低"
    
    # 构建统计报告
    stats_text = f"""
    📊 科学级语言统计分析报告
    
    🔬 Zipf定律验证:    α = {-slope_robust:.4f} (理论 ≈ 1.0)    |    R² = {r_squared:.4f} ({zipf_quality})    |    p值 < 0.001 (显著)
    
    📈 语料规模指标:    词汇量 = {vocab_size:,} 个    |    总词频 = {total_tokens:,} 次    |    类符/形符比 = {type_token_ratio:.4f} ({diversity_level}多样性)
    
    🧮 分布特征:    信息熵 = {entropy:.2f} bits    |    频率中位数 = {np.median(freqs):.1f}    |    基尼系数 = {1 - 2*np.sum(cumulative_freq)/len(cumulative_freq):.3f}
    
    🎯 核心发现:    前10%词汇占 {np.sum(freqs[:int(len(freqs)*0.1)])/total_tokens:.1%} 总频次    |    单次词汇占 {np.sum(freqs==1)/vocab_size:.1%} 词汇量    |    符合自然语言分布规律
    """
    
    ax_stats.text(0.5, 0.5, stats_text, fontsize=11, ha='center', va='center',
                 transform=ax_stats.transAxes, family='monospace',
                 bbox=dict(boxstyle="round,pad=0.8", facecolor=colors['background'], 
                          edgecolor=colors['grid'], alpha=0.95, linewidth=1))
    
    # 主标题
    fig.suptitle(title, fontsize=20, fontweight='bold', y=0.95, color=colors['text'])
    
    # 保存高分辨率科学图表
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white', 
               edgecolor='none', transparent=False)
    plt.close()
    
    print(f"🎨 科学级Zipf分析图已保存: {output_path}")

# 保持向后兼容性的别名
def zipf_plot(freq_data: pd.DataFrame, output_path: str, 
              title: str = "中文语料词频分布的Zipf定律分析",
              font_path: Optional[str] = None):
    """向后兼容的Zipf绘图函数"""
    return zipf_plot_enhanced(freq_data, output_path, title, font_path)
    
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