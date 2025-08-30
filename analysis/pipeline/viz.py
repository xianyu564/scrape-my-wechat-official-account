#!/usr/bin/env python3
"""
Visualization module: charts and word clouds with scientific styling
"""

import os
import warnings
from collections import Counter
from typing import Any, Dict, List, Optional

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import font_manager
from wordcloud import WordCloud
from functools import lru_cache
import time
from pathlib import Path


# Set up matplotlib for Chinese font support
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans', 'Liberation Sans']
plt.rcParams['axes.unicode_minus'] = False

# Scientific color schemes
COLOR_SCHEMES = {
    'nature': ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b'],
    'science': ['#2E8B57', '#4682B4', '#CD853F', '#DC143C', '#9932CC', '#FF8C00'],
    'calm': ['#6B8E23', '#4F94CD', '#CD853F', '#F4A460', '#9ACD32', '#87CEEB'],
    'muted': ['#88CCEE', '#CC6677', '#DDCC77', '#117733', '#332288', '#AA4499'],
    'solar': ['#FDB863', '#E08214', '#8073AC', '#542788', '#2D004B', '#B35806']
}


@lru_cache(maxsize=4)
def _find_system_chinese_fonts() -> List[str]:
    """
    Cache-optimized system Chinese font detection
    
    Returns:
        List[str]: Available Chinese font paths
    """
    print("🔍 Scanning system fonts for Chinese support...")
    
    # Common Chinese font names to search for
    chinese_font_names = [
        'SimHei', 'SimSun', 'Microsoft YaHei', 'PingFang SC', 'Hiragino Sans GB',
        'STHeiti', 'STSong', 'Source Han Sans', 'Noto Sans CJK', 'WenQuanYi',
        'AR PL UMing', 'AR PL UKai', 'Liberation Sans', 'DejaVu Sans'
    ]
    
    # System font directories to search
    font_dirs = [
        '/System/Library/Fonts',  # macOS
        '/usr/share/fonts',       # Linux
        '/usr/local/share/fonts', # Linux (user)
        '/Library/Fonts',         # macOS (user)
        'C:/Windows/Fonts',       # Windows
        '/opt/homebrew/share/fonts',  # macOS (Homebrew)
        '~/.fonts',               # Linux (user)
    ]
    
    found_fonts = []
    
    # Search system font directories
    for font_dir in font_dirs:
        font_path = Path(font_dir).expanduser()
        if font_path.exists():
            for font_file in font_path.rglob('*.ttf'):
                if any(name.lower() in font_file.name.lower() for name in chinese_font_names):
                    found_fonts.append(str(font_file))
            for font_file in font_path.rglob('*.otf'):
                if any(name.lower() in font_file.name.lower() for name in chinese_font_names):
                    found_fonts.append(str(font_file))
    
    # Also check matplotlib's font cache
    try:
        for font in font_manager.fontManager.ttflist:
            font_name = font.name.lower()
            if any(chinese_name.lower() in font_name for chinese_name in chinese_font_names):
                if font.fname not in found_fonts:
                    found_fonts.append(font.fname)
    except:
        pass
    
    print(f"✅ Found {len(found_fonts)} potential Chinese fonts")
    return found_fonts[:10]  # Return top 10 candidates


@lru_cache(maxsize=2)
def setup_chinese_font(font_path: Optional[str] = None) -> str:
    """
    Setup Chinese font for matplotlib and wordcloud with comprehensive auto-detection and caching
    
    Args:
        font_path: Path to Chinese font file
        
    Returns:
        str: Font path for wordcloud
    """
    if font_path and os.path.exists(font_path):
        try:
            font_prop = font_manager.FontProperties(fname=font_path)
            plt.rcParams['font.family'] = font_prop.get_name()
            print(f"✅ Using custom font: {font_path}")
            return font_path
        except Exception as e:
            warnings.warn(f"Failed to load custom font {font_path}: {e}")
    
    # First, try to find fonts using cached font detection
    try:
        font_candidates = _find_system_chinese_fonts()
        
        for font_file in font_candidates:
            try:
                # Test if font can be loaded
                font_prop = font_manager.FontProperties(fname=font_file)
                plt.rcParams['font.family'] = font_prop.get_name()
                print(f"✅ Using Chinese font file: {font_file}")
                return font_file
            except Exception:
                continue
                
    except Exception as e:
        warnings.warn(f"Font detection failed: {e}")
    
    # Fallback to matplotlib's default handling
    print("⚠️  Using system default font (Chinese characters may not display properly)")
    return ""

    try:
        # Check for installed fonts that support Chinese
        chinese_font_families = [
            'Noto Sans CJK SC', 'Noto Sans CJK TC', 'Noto Sans CJK',
            'Source Han Sans SC', 'Source Han Sans TC', 'Source Han Sans',
            'Microsoft YaHei', 'SimHei', 'SimSun', 'STHeiti',
            'PingFang SC', 'Hiragino Sans GB', 'Heiti SC',
            'WenQuanYi Micro Hei', 'WenQuanYi Zen Hei', 'AR PL UMing CN'
        ]

        available_fonts = set()
        for font in font_manager.fontManager.ttflist:
            if font.name:
                available_fonts.add(font.name)

        for font_family in chinese_font_families:
            if font_family in available_fonts:
                plt.rcParams['font.family'] = font_family
                # Try to find the actual font file for wordcloud
                for font in font_manager.fontManager.ttflist:
                    if font.name == font_family:
                        print(f"✅ Using system font: {font_family} ({font.fname})")
                        return font.fname
                print(f"✅ Using system font: {font_family}")
                return None  # Font available but no file path found
    except Exception as e:
        warnings.warn(f"Font manager search failed: {e}")

    # Fallback: Try to find font files in common locations
    chinese_fonts = [
        # Linux - Noto fonts
        '/usr/share/fonts/truetype/noto/NotoSansCJK-Regular.ttc',
        '/usr/share/fonts/truetype/noto/NotoSansCJK-Bold.ttc',
        '/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc',
        '/usr/share/fonts/noto-cjk/NotoSansCJK-Regular.ttc',
        # Linux - Other fonts
        '/usr/share/fonts/truetype/droid/DroidSansFallbackFull.ttf',
        '/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf',
        '/usr/share/fonts/truetype/wqy/wqy-microhei.ttc',
        '/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc',
        '/usr/share/fonts/truetype/arphic/uming.ttc',
        # macOS
        '/System/Library/Fonts/PingFang.ttc',
        '/System/Library/Fonts/STHeiti Light.ttc',
        '/System/Library/Fonts/Hiragino Sans GB.ttc',
        '/Library/Fonts/Arial Unicode MS.ttf',
        # Windows
        'C:\\Windows\\Fonts\\simhei.ttf',
        'C:\\Windows\\Fonts\\simsun.ttc',
        'C:\\Windows\\Fonts\\msyh.ttc',  # Microsoft YaHei
        'C:\\Windows\\Fonts\\msyhbd.ttc',  # Microsoft YaHei Bold
    ]

    for font_file in chinese_fonts:
        if os.path.exists(font_file):
            try:
                font_prop = font_manager.FontProperties(fname=font_file)
                plt.rcParams['font.family'] = font_prop.get_name()
                print(f"✅ Using Chinese font file: {font_file}")
                return font_file
            except Exception as e:
                warnings.warn(f"Failed to load font {font_file}: {e}")
                continue

    # Final fallback: try downloading Noto fonts if none found
    print("⚠️  No Chinese font found locally")
    print("💡 Consider installing Noto CJK fonts:")
    print("   - Ubuntu/Debian: sudo apt install fonts-noto-cjk")
    print("   - CentOS/RHEL: sudo yum install google-noto-cjk-fonts")
    print("   - macOS: brew install --cask font-noto-sans-cjk-sc")
    print("   - Or download from: https://fonts.google.com/noto/specimen/Noto+Sans+SC")

    return None


def create_zipf_panels(frequencies: Counter,
                      output_path: str,
                      zipf_results: Dict[str, float],
                      font_path: Optional[str] = None,
                      color_scheme: str = 'nature') -> None:
    """
    Create four-panel Zipf analysis plot
    
    Args:
        frequencies: Term frequency counter
        output_path: Output image path
        zipf_results: Zipf analysis results
        font_path: Chinese font path
        color_scheme: Color scheme name
    """
    setup_chinese_font(font_path)
    colors = COLOR_SCHEMES.get(color_scheme, COLOR_SCHEMES['nature'])

    # Prepare data
    sorted_freqs = sorted(frequencies.values(), reverse=True)
    ranks = np.arange(1, len(sorted_freqs) + 1)
    freqs = np.array(sorted_freqs)

    # Filter positive frequencies for log scale
    valid_indices = freqs > 0
    ranks = ranks[valid_indices]
    freqs = freqs[valid_indices]

    if len(ranks) < 10:
        print("⚠️  Insufficient data for Zipf analysis")
        return

    # Create 2x2 subplot
    fig, axes = plt.subplots(2, 2, figsize=(12, 10), dpi=300)
    fig.suptitle('Zipf\'s Law Analysis - Scientific Grade', fontsize=16, fontweight='bold')

    # Panel 1: Log-log rank-frequency plot
    ax1 = axes[0, 0]
    ax1.loglog(ranks, freqs, 'o', color=colors[0], alpha=0.7, markersize=4)

    # Add fitted line
    log_ranks = np.log(ranks)
    log_freqs = np.log(freqs)
    slope = zipf_results.get('slope', -1)
    intercept = zipf_results.get('intercept', 0)

    fitted_freqs = np.exp(intercept + slope * log_ranks)
    ax1.loglog(ranks, fitted_freqs, '-', color=colors[1], linewidth=2,
               label=f'Fitted: slope={slope:.3f}')

    ax1.set_xlabel('Rank (log scale)')
    ax1.set_ylabel('Frequency (log scale)')
    ax1.set_title('Rank-Frequency Relationship')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Panel 2: Residuals plot
    ax2 = axes[0, 1]
    residuals = log_freqs - (intercept + slope * log_ranks)
    ax2.scatter(log_ranks, residuals, color=colors[2], alpha=0.6, s=20)
    ax2.axhline(y=0, color='red', linestyle='--', alpha=0.8)
    ax2.set_xlabel('Log Rank')
    ax2.set_ylabel('Residuals')
    ax2.set_title('Residual Analysis')
    ax2.grid(True, alpha=0.3)

    # Panel 3: Cumulative distribution
    ax3 = axes[1, 0]
    cumulative_freq = np.cumsum(freqs) / np.sum(freqs)
    ax3.plot(ranks, cumulative_freq, '-', color=colors[3], linewidth=2)
    ax3.set_xlabel('Rank')
    ax3.set_ylabel('Cumulative Frequency Proportion')
    ax3.set_title('Cumulative Distribution')
    ax3.grid(True, alpha=0.3)

    # Panel 4: Frequency histogram
    ax4 = axes[1, 1]
    ax4.hist(freqs, bins=50, color=colors[4], alpha=0.7, edgecolor='black', linewidth=0.5)
    ax4.set_xlabel('Frequency')
    ax4.set_ylabel('Number of Terms')
    ax4.set_title('Frequency Distribution')
    ax4.set_yscale('symlog', linthresh=1)
    ax4.grid(True, alpha=0.3)

    # Add statistics text
    r_squared = zipf_results.get('r_squared', 0)
    stats_text = f"R² = {r_squared:.3f}\nSlope = {slope:.3f}\nTerms = {len(frequencies):,}"
    fig.text(0.02, 0.02, stats_text, fontsize=10,
             bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgray", alpha=0.7))
    

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()

    print(f"📈 Zipf panels saved: {output_path}")
    print("ℹ️  Note: Zipf law analysis is an approximation - please interpret results carefully")


def create_heaps_plot(corpus_tokens: List[List[str]],
                     output_path: str,
                     heaps_results: Dict[str, float],
                     font_path: Optional[str] = None,
                     color_scheme: str = 'nature') -> None:
    """
    Create Heaps' law analysis plot with bootstrap confidence intervals
    
    Args:
        corpus_tokens: Tokenized documents
        output_path: Output image path
        heaps_results: Heaps analysis results
        font_path: Chinese font path
        color_scheme: Color scheme name
    """
    setup_chinese_font(font_path)
    colors = COLOR_SCHEMES.get(color_scheme, COLOR_SCHEMES['nature'])

    # Calculate vocabulary growth
    vocabulary = set()
    corpus_sizes = []
    vocab_sizes = []

    total_tokens = 0
    for tokens in corpus_tokens:
        total_tokens += len(tokens)
        vocabulary.update(tokens)
        corpus_sizes.append(total_tokens)
        vocab_sizes.append(len(vocabulary))
    
    # Adaptive handling for small datasets
    min_points = min(10, max(3, len(corpus_sizes) // 2))
    if len(corpus_sizes) < min_points:
        print(f"⚠️  Insufficient data for Heaps analysis ({len(corpus_sizes)} points)")

        return

    n = np.array(corpus_sizes)
    V = np.array(vocab_sizes)

    # Create plot
    fig, ax = plt.subplots(1, 1, figsize=(10, 6), dpi=300)

    # Plot actual data
    ax.plot(n, V, 'o', color=colors[0], alpha=0.7, markersize=4, label='Observed Data')
    
    # Plot fitted curve and confidence bands
    K = heaps_results.get('K', 0)
    beta = heaps_results.get('beta', 0)
    confidence_lower = heaps_results.get('confidence_lower', 0)
    confidence_upper = heaps_results.get('confidence_upper', 0)
    

    if K > 0 and beta > 0:
        fitted_V = K * (n ** beta)
        ax.plot(n, fitted_V, '-', color=colors[1], linewidth=2,
                label=f'Heaps Law: V = {K:.1f} × n^{beta:.3f}')
        
        # Add bootstrap confidence bands if available
        if confidence_lower > 0 and confidence_upper > 0:
            upper_V = K * (n ** confidence_upper)
            lower_V = K * (n ** confidence_lower)
            ax.fill_between(n, lower_V, upper_V, alpha=0.2, color=colors[1],
                          label=f'95% Confidence Band (β: {confidence_lower:.3f}-{confidence_upper:.3f})')
    

    ax.set_xlabel('Corpus Size (tokens)')
    ax.set_ylabel('Vocabulary Size (unique tokens)')
    ax.set_title('Heaps\' Law: Vocabulary Growth Analysis')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Add statistics with warning if applicable
    r_squared = heaps_results.get('r_squared', 0)
    valid_points = heaps_results.get('valid_points', len(corpus_sizes))
    warning = heaps_results.get('warning', '')
    
    stats_text = f"K = {K:.1f}\nβ = {beta:.3f}\nR² = {r_squared:.3f}\nPoints = {valid_points}"
    if warning:
        stats_text += f"\n⚠️  {warning[:30]}..."  # Truncate long warnings
    
    ax.text(0.05, 0.95, stats_text, transform=ax.transAxes, fontsize=10,
            verticalalignment='top', bbox=dict(boxstyle="round,pad=0.3", 
                                               facecolor="lightyellow" if warning else "lightgray", 
                                               alpha=0.8))
    
    # Add approximation note
    ax.text(0.05, 0.02, "Note: Heaps law fitting is an approximation - interpret results carefully",
            transform=ax.transAxes, fontsize=8, style='italic', alpha=0.7)
    

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()

    print(f"📈 Heaps plot saved: {output_path}")
    if warning:
        print(f"⚠️  {warning}")


def create_wordcloud(frequencies: Counter,
                    output_path: str,
                    title: str = "Word Cloud",
                    font_path: Optional[str] = None,
                    max_words: int = 200,
                    color_scheme: str = 'nature',
                    mask_path: Optional[str] = None) -> None:
    """
    Create beautiful word cloud with CJK font support
    
    Args:
        frequencies: Term frequencies
        output_path: Output image path
        title: Word cloud title
        font_path: Chinese font path
        max_words: Maximum words to display
        color_scheme: Color scheme name
        mask_path: Path to mask image
    """
    if not frequencies:
        print("⚠️  No frequencies for word cloud")
        return

    # Setup font
    font_path = setup_chinese_font(font_path)

    # Color scheme
    colors = COLOR_SCHEMES.get(color_scheme, COLOR_SCHEMES['nature'])

    def color_func(word, font_size, position, orientation, random_state=None, **kwargs):
        """Custom color function for scientific aesthetics"""
        color_idx = hash(word) % len(colors)
        return colors[color_idx]

    # Load mask if provided
    mask = None
    if mask_path and os.path.exists(mask_path):
        from PIL import Image
        try:
            mask = np.array(Image.open(mask_path))
        except Exception as e:
            warnings.warn(f"Failed to load mask {mask_path}: {e}")

    # Create WordCloud
    wordcloud_params = {
        'width': 800,
        'height': 600,
        'max_words': max_words,
        'relative_scaling': 0.5,
        'min_font_size': 10,
        'background_color': 'white',
        'color_func': color_func,
        'random_state': 42,  # For reproducibility
        'collocations': False,
        'prefer_horizontal': 0.7
    }

    if font_path:
        wordcloud_params['font_path'] = font_path

    if mask is not None:
        wordcloud_params['mask'] = mask

    try:
        wordcloud = WordCloud(**wordcloud_params)
        wordcloud.generate_from_frequencies(frequencies)

        # Create figure
        fig, ax = plt.subplots(1, 1, figsize=(12, 8), dpi=300)
        ax.imshow(wordcloud, interpolation='bilinear')
        ax.axis('off')
        ax.set_title(title, fontsize=16, fontweight='bold', pad=20)

        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()

        print(f"🎨 Word cloud saved: {output_path}")

    except Exception as e:
        warnings.warn(f"Word cloud generation failed: {e}")


def create_yearly_comparison_chart(freq_by_year: Dict[str, Counter],
                                 output_path: str,
                                 top_n: int = 20,
                                 font_path: Optional[str] = None,
                                 color_scheme: str = 'nature') -> None:
    """
    Create yearly word frequency comparison bar chart
    
    Args:
        freq_by_year: Frequencies by year
        output_path: Output image path
        top_n: Top N words to show
        font_path: Chinese font path
        color_scheme: Color scheme name
    """
    setup_chinese_font(font_path)
    colors = COLOR_SCHEMES.get(color_scheme, COLOR_SCHEMES['nature'])

    if not freq_by_year:
        print("⚠️  No yearly data for comparison chart")
        return

    years = sorted(freq_by_year.keys())
    if len(years) < 2:
        print("⚠️  Need at least 2 years for comparison")
        return

    # Get overall top words
    overall_freq = Counter()
    for year_freq in freq_by_year.values():
        overall_freq.update(year_freq)

    top_words = [word for word, _ in overall_freq.most_common(top_n)]

    # Prepare data for plotting
    data = []
    for year in years:
        year_freq = freq_by_year[year]
        for word in top_words:
            data.append({
                'year': year,
                'word': word,
                'frequency': year_freq.get(word, 0)
            })

    # Create horizontal bar chart
    fig, ax = plt.subplots(1, 1, figsize=(12, max(8, len(top_words) * 0.4)), dpi=300)

    # Plot bars for each year
    y_positions = np.arange(len(top_words))
    bar_width = 0.8 / len(years)

    for i, year in enumerate(years):
        frequencies = [freq_by_year[year].get(word, 0) for word in top_words]
        color = colors[i % len(colors)]

        bars = ax.barh(y_positions + i * bar_width, frequencies,
                      bar_width, label=year, color=color, alpha=0.8)

        # Add frequency labels
        for j, (bar, freq) in enumerate(zip(bars, frequencies)):
            if freq > 0:
                ax.text(bar.get_width() + max(frequencies) * 0.01,
                       bar.get_y() + bar.get_height()/2,
                       f'{freq}', ha='left', va='center', fontsize=8)

    ax.set_yticks(y_positions + bar_width * (len(years) - 1) / 2)
    ax.set_yticklabels(top_words)
    ax.set_xlabel('Frequency')
    ax.set_title(f'Top {top_n} Words by Year', fontsize=14, fontweight='bold')
    ax.legend(title='Year', bbox_to_anchor=(1.05, 1), loc='upper left')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()

    print(f"📊 Yearly comparison chart saved: {output_path}")


def create_growth_chart(growth_data: List[Dict[str, Any]],
                       output_path: str,
                       top_n: int = 20,
                       font_path: Optional[str] = None,
                       color_scheme: str = 'nature') -> None:
    """
    Create year-over-year growth chart
    
    Args:
        growth_data: YoY growth data
        output_path: Output image path
        top_n: Top N growing words
        font_path: Chinese font path
        color_scheme: Color scheme name
    """
    setup_chinese_font(font_path)
    colors = COLOR_SCHEMES.get(color_scheme, COLOR_SCHEMES['nature'])

    if not growth_data:
        print("⚠️  No growth data for chart")
        return

    # Sort by growth and take top N
    sorted_growth = sorted(growth_data, key=lambda x: x['growth'], reverse=True)[:top_n]

    words = [item['word'] for item in sorted_growth]
    growth_values = [item['growth'] for item in sorted_growth]

    # Create horizontal bar chart
    fig, ax = plt.subplots(1, 1, figsize=(10, max(6, len(words) * 0.3)), dpi=300)

    bars = ax.barh(range(len(words)), growth_values, color=colors[0], alpha=0.8)

    # Color bars by growth (positive green, negative red)
    for bar, growth in zip(bars, growth_values):
        if growth > 0:
            bar.set_color('#2ca02c')  # Green for positive
        else:
            bar.set_color('#d62728')  # Red for negative

    # Add value labels
    for i, (bar, growth) in enumerate(zip(bars, growth_values)):
        ax.text(bar.get_width() + max(growth_values) * 0.01,
               bar.get_y() + bar.get_height()/2,
               f'+{growth}' if growth > 0 else str(growth),
               ha='left', va='center', fontsize=10)

    ax.set_yticks(range(len(words)))
    ax.set_yticklabels(words)
    ax.set_xlabel('Frequency Growth')
    ax.set_title(f'Top {top_n} Year-over-Year Word Growth', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()

    print(f"📈 Growth chart saved: {output_path}")
