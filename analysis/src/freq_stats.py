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
              title: str = "词频分布的 Zipf 定律分析",
              font_path: Optional[str] = None):
    """
    绘制 Zipf 定律分析图
    
    Args:
        freq_data: 词频数据，需包含 'freq' 列
        output_path: 输出图片路径
        title: 图表标题
        font_path: 中文字体路径
    """
    if freq_data.empty:
        warnings.warn("词频数据为空，无法绘制 Zipf 图")
        return
    
    # 设置中文字体
    if font_path and os.path.exists(font_path):
        font_prop = font_manager.FontProperties(fname=font_path)
        plt.rcParams['font.family'] = font_prop.get_name()
    else:
        # 尝试系统默认中文字体
        try:
            plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
        except:
            pass
    
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
    
    # 创建图表
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # 左图：双对数散点图
    ax1.loglog(ranks, freqs, 'bo', alpha=0.6, markersize=3)
    ax1.set_xlabel('词频排名 (log)')
    ax1.set_ylabel('词频 (log)')
    ax1.set_title('Zipf 定律双对数图')
    ax1.grid(True, alpha=0.3)
    
    # 拟合直线
    log_ranks = np.log(ranks)
    log_freqs = np.log(freqs)
    
    # 线性回归
    coeff = np.polyfit(log_ranks, log_freqs, 1)
    slope, intercept = coeff
    
    # 绘制拟合线
    fitted_freqs = np.exp(intercept) * ranks ** slope
    ax1.loglog(ranks, fitted_freqs, 'r-', linewidth=2, 
               label=f'拟合线: y = {np.exp(intercept):.2f} * x^{slope:.2f}')
    ax1.legend()
    
    # 计算 R²
    ss_res = np.sum((log_freqs - (slope * log_ranks + intercept)) ** 2)
    ss_tot = np.sum((log_freqs - np.mean(log_freqs)) ** 2)
    r_squared = 1 - (ss_res / ss_tot)
    
    # 右图：残差分析
    residuals = log_freqs - (slope * log_ranks + intercept)
    ax2.scatter(log_ranks, residuals, alpha=0.6, s=10)
    ax2.axhline(y=0, color='r', linestyle='--', alpha=0.8)
    ax2.set_xlabel('log(排名)')
    ax2.set_ylabel('残差')
    ax2.set_title(f'残差分析 (R² = {r_squared:.3f})')
    ax2.grid(True, alpha=0.3)
    
    # 总标题
    fig.suptitle(title, fontsize=16, y=0.98)
    
    # 添加说明文字
    zipf_explanation = (
        f"Zipf 定律: 词频 ∝ 1/排名^α\n"
        f"拟合斜率: {slope:.3f} (理论值 ≈ -1)\n"
        f"拟合优度: R² = {r_squared:.3f}\n"
        f"总词数: {len(freqs):,}, 总频次: {freqs.sum():,}"
    )
    
    fig.text(0.02, 0.02, zipf_explanation, fontsize=10,
             bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgray", alpha=0.8))
    
    plt.tight_layout()
    
    # 保存图片
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Zipf 分析图已保存: {output_path}")
    print(f"拟合参数: 斜率={slope:.3f}, R²={r_squared:.3f}")


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