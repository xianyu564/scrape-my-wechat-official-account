"""
词云可视化模块 - 生成高颜值中文词云
"""

import os
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from typing import Dict, Optional, List, Tuple
import warnings


def create_circular_mask(size: Tuple[int, int] = (800, 800)) -> np.ndarray:
    """
    创建圆形蒙版
    
    Args:
        size: 蒙版尺寸 (width, height)
    
    Returns:
        np.ndarray: 蒙版数组，白色为有效区域，黑色为遮罩区域
    """
    width, height = size
    
    # 创建白色背景
    img = Image.new('RGB', (width, height), 'white')
    draw = ImageDraw.Draw(img)
    
    # 绘制黑色圆形（作为遮罩）
    margin = min(width, height) // 10
    draw.ellipse([margin, margin, width-margin, height-margin], fill='black')
    
    # 转换为数组
    mask = np.array(img)
    
    return mask


def create_heart_mask(size: Tuple[int, int] = (800, 800)) -> np.ndarray:
    """
    创建心形蒙版
    
    Args:
        size: 蒙版尺寸 (width, height)
    
    Returns:
        np.ndarray: 心形蒙版数组
    """
    width, height = size
    img = Image.new('RGB', (width, height), 'white')
    draw = ImageDraw.Draw(img)
    
    # 心形路径计算
    center_x, center_y = width // 2, height // 2
    scale = min(width, height) // 3
    
    points = []
    for i in range(360):
        t = np.radians(i)
        x = 16 * np.sin(t)**3
        y = -(13 * np.cos(t) - 5 * np.cos(2*t) - 2 * np.cos(3*t) - np.cos(4*t))
        points.append((center_x + x * scale // 16, center_y + y * scale // 16))
    
    draw.polygon(points, fill='black')
    
    return np.array(img)


def scientific_color_func(word: str, font_size: int, position: Tuple[int, int], 
                         orientation: int, random_state: Optional[int] = None,
                         **kwargs) -> str:
    """
    科学研究风格配色函数 - 优雅、专业、符合学术审美
    
    Args:
        word: 词汇
        font_size: 字体大小
        position: 位置
        orientation: 方向
        random_state: 随机种子
    
    Returns:
        str: 颜色十六进制代码
    """
    np.random.seed(random_state)
    
    # 科学研究专用配色方案 - 基于Nature, Science等期刊的配色
    scientific_palette = {
        'primary': ['#2E86AB', '#A23B72', '#F18F01', '#C73E1D'],      # 主色调：蓝、紫、橙、红
        'secondary': ['#5D737E', '#64A6BD', '#90A959', '#D07A56'],    # 次色调：灰蓝、亮蓝、绿、橘
        'accent': ['#8E44AD', '#2980B9', '#27AE60', '#F39C12']        # 强调色：紫、蓝、绿、黄
    }
    
    # 根据字体大小确定颜色类别和饱和度
    if font_size > 60:
        # 超大字体：使用主色调，高饱和度
        colors = scientific_palette['primary']
        saturation = 0.9
    elif font_size > 35:
        # 大字体：使用次色调，中等饱和度
        colors = scientific_palette['secondary'] 
        saturation = 0.8
    else:
        # 小字体：使用强调色，适中饱和度
        colors = scientific_palette['accent']
        saturation = 0.7
    
    # 基于词汇特征选择颜色（确保一致性）
    word_hash = hash(word) % len(colors)
    base_color = colors[word_hash]
    
    # 转换为HSV进行亮度调整
    rgb = mcolors.hex2color(base_color)
    hsv = mcolors.rgb_to_hsv(rgb)
    
    # 微调饱和度和亮度
    hsv[1] = min(1.0, hsv[1] * saturation)  # 调整饱和度
    hsv[2] = max(0.3, min(0.9, hsv[2] + np.random.normal(0, 0.1)))  # 轻微随机化亮度
    
    # 转换回RGB和hex
    rgb_adjusted = mcolors.hsv_to_rgb(hsv)
    return mcolors.rgb2hex(rgb_adjusted)


def create_wordcloud(word_freq: Dict[str, int],
                    output_path: str,
                    title: str = "",
                    mask_path: Optional[str] = None,
                    font_path: Optional[str] = None,
                    colormap: str = "viridis",
                    width: int = 1600,
                    height: int = 1200,
                    background_color: str = "white",
                    max_words: int = 200,
                    use_scientific_colors: bool = True) -> bool:
    """
    生成高质量科学研究风格词云图
    
    Args:
        word_freq: 词频字典 {word: frequency}
        output_path: 输出文件路径
        title: 图片标题
        mask_path: 蒙版图片路径
        font_path: 中文字体路径
        colormap: 颜色映射（当不使用科学配色时）
        width: 图片宽度
        height: 图片高度
        background_color: 背景颜色
        max_words: 最大词数
        use_scientific_colors: 是否使用科学研究配色
    
    Returns:
        bool: 是否成功生成
    """
    if not word_freq:
        warnings.warn("词频数据为空，无法生成词云")
        return False
    
    try:
        # 加载蒙版
        mask = None
        if mask_path and os.path.exists(mask_path):
            mask_img = Image.open(mask_path)
            mask = np.array(mask_img)
        
        # 配置词云参数 - 优化为高质量科学图表
        wc_params = {
            'width': width,
            'height': height,
            'background_color': background_color,
            'max_words': max_words,
            'relative_scaling': 0.6,          # 增加相对缩放，层次更明显
            'min_font_size': 14,              # 提高最小字体
            'max_font_size': 120,             # 提高最大字体
            'prefer_horizontal': 0.8,         # 更多水平文字，便于阅读
            'random_state': 42,
            'collocations': False,            # 避免词汇搭配重复
            'include_numbers': False,         # 不包含纯数字
            'normalize_plurals': False        # 保持中文原样
        }
        
        # 设置蒙版
        if mask is not None:
            wc_params['mask'] = mask
        
        # 设置字体
        if font_path and os.path.exists(font_path):
            wc_params['font_path'] = font_path
        
        # 设置配色方案
        if use_scientific_colors:
            wc_params['color_func'] = scientific_color_func
        else:
            wc_params['colormap'] = colormap
        
        # 创建词云对象
        wordcloud = WordCloud(**wc_params)
        
        # 生成词云
        wordcloud.generate_from_frequencies(word_freq)
        
        # 创建高质量图形
        plt.figure(figsize=(width/100, height/100), dpi=150)
        plt.imshow(wordcloud, interpolation='bilinear')
        plt.axis('off')
        
        # 添加科学风格标题
        if title:
            plt.title(title, fontsize=20, pad=30, fontweight='bold', 
                     color='#2C3E50', family='serif')
        
        # 添加优雅的统计信息标注
        stats_text = f"词汇量: {len(word_freq):,} | 总词频: {sum(word_freq.values()):,} | 词汇密度: {len(word_freq)/sum(word_freq.values()):.3f}"
        plt.figtext(0.02, 0.02, stats_text, fontsize=11, 
                   color='#34495E', style='italic',
                   bbox=dict(boxstyle="round,pad=0.4", facecolor="white", 
                            edgecolor='#BDC3C7', alpha=0.9))
        
        # 保存高质量图片
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        plt.savefig(output_path, dpi=300, bbox_inches='tight', 
                   facecolor='white', edgecolor='none', 
                   pad_inches=0.2)  # 添加小边距
        plt.close()
        
        print(f"🎨 高质量词云已保存: {output_path}")
        return True
        
    except Exception as e:
        warnings.warn(f"生成词云失败: {e}")
        return False


def generate_overall_wordcloud(freq_data: pd.DataFrame,
                              output_dir: str = "analysis/out",
                              mask_path: Optional[str] = None,
                              font_path: Optional[str] = None,
                              top_n: int = 200) -> bool:
    """
    生成整体词云
    
    Args:
        freq_data: 词频数据，包含 'word' 和 'freq' 列
        output_dir: 输出目录
        mask_path: 蒙版路径
        font_path: 字体路径
        top_n: 使用前 N 个词
    
    Returns:
        bool: 是否成功
    """
    if freq_data.empty:
        warnings.warn("词频数据为空")
        return False
    
    try:
        # 取前 N 个词
        top_words = freq_data.head(top_n)
        
        # 确保有数据
        if top_words.empty:
            warnings.warn("没有词频数据")
            return False
        
        # 确保频率列存在且为数值类型
        if 'freq' not in top_words.columns:
            warnings.warn("词频数据中缺少'freq'列")
            return False
            
        # 过滤出有效数据 - 避免类型转换问题
        valid_indices = []
        for idx, row in top_words.iterrows():
            try:
                freq_val = row['freq']
                if pd.notna(freq_val) and (isinstance(freq_val, (int, float)) and freq_val > 0):
                    valid_indices.append(idx)
            except:
                continue
        
        if not valid_indices:
            warnings.warn("没有有效的词频数据")
            return False
            
        valid_data = top_words.loc[valid_indices]
        
        word_freq = {}
        for _, row in valid_data.iterrows():
            try:
                word = str(row['word']).strip()
                freq_val = row['freq']
                
                # 简化类型转换
                if isinstance(freq_val, (int, float)) and not pd.isna(freq_val):
                    freq = int(freq_val)
                    if freq > 0 and word:
                        word_freq[word] = freq
                        
            except (ValueError, TypeError, AttributeError) as e:
                print(f"⚠️ 跳过无效数据: word={row.get('word', 'N/A')}, freq={row.get('freq', 'N/A')}, error={e}")
                continue
        
        if not word_freq:
            warnings.warn("处理后没有有效的词频数据")
            return False
        
        # 生成词云
        output_path = os.path.join(output_dir, "wordcloud_overall.png")
        title = f"整体词云 (Top {len(word_freq)} 词汇)"
        
        return create_wordcloud(
            word_freq=word_freq,
            output_path=output_path,
            title=title,
            mask_path=mask_path,
            font_path=font_path,
            use_scientific_colors=True
        )
    except Exception as e:
        print(f"❌ 整体词云生成失败的详细错误:")
        print(f"   错误类型: {type(e).__name__}")
        print(f"   错误信息: {str(e)}")
        print(f"   数据统计: 有效词汇数={len(word_freq) if 'word_freq' in locals() else '未知'}")
        if 'word_freq' in locals() and word_freq:
            sample_items = list(word_freq.items())[:3]
            print(f"   样本数据: {sample_items}")
        import traceback
        traceback.print_exc()
        warnings.warn(f"生成整体词云失败: {e}")
        return False


def generate_yearly_wordclouds(freq_by_year: pd.DataFrame,
                              output_dir: str = "analysis/out",
                              mask_path: Optional[str] = None,
                              font_path: Optional[str] = None,
                              top_n: int = 100) -> List[str]:
    """
    生成年度词云
    
    Args:
        freq_by_year: 年度词频数据，包含 'year', 'word', 'freq' 列
        output_dir: 输出目录
        mask_path: 蒙版路径
        font_path: 字体路径
        top_n: 每年使用前 N 个词
    
    Returns:
        List[str]: 生成的文件路径列表
    """
    if freq_by_year.empty:
        warnings.warn("年度词频数据为空")
        return []
    
    generated_files = []
    years = sorted(freq_by_year['year'].unique())
    
    for year in years:
        # 获取该年数据
        year_data = freq_by_year[freq_by_year['year'] == year]
        
        if year_data.empty:
            continue
        
        # 取前 N 个词
        top_words = year_data.head(top_n)
        
        if top_words.empty:
            continue
        
        # 安全地处理词频数据
        word_freq = {}
        for _, row in top_words.iterrows():
            try:
                word = str(row['word']).strip()
                freq_val = row['freq']
                
                # 简化类型转换
                if isinstance(freq_val, (int, float)) and not pd.isna(freq_val):
                    freq = int(freq_val)
                    if freq > 0 and word:
                        word_freq[word] = freq
                        
            except (ValueError, TypeError, AttributeError) as e:
                print(f"⚠️ 跳过年度数据: year={year}, word={row.get('word', 'N/A')}, freq={row.get('freq', 'N/A')}, error={e}")
                continue
        
        if not word_freq:
            continue
        
        # 生成词云
        output_path = os.path.join(output_dir, f"wordcloud_{year}.png")
        title = f"{year} 年度词云 (Top {len(word_freq)} 词汇)"
        
        success = create_wordcloud(
            word_freq=word_freq,
            output_path=output_path,
            title=title,
            mask_path=mask_path,
            font_path=font_path,
            use_scientific_colors=True
        )
        
        if success:
            generated_files.append(output_path)
    
    return generated_files


def create_default_mask(output_path: str = "analysis/assets/mask.png",
                       shape: str = "circle",
                       size: Tuple[int, int] = (800, 800)) -> bool:
    """
    创建默认蒙版文件
    
    Args:
        output_path: 输出文件路径
        shape: 形状类型 ('circle' 或 'heart')
        size: 尺寸
    
    Returns:
        bool: 是否成功创建
    """
    try:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        if shape == "circle":
            mask = create_circular_mask(size)
        elif shape == "heart":
            mask = create_heart_mask(size)
        else:
            raise ValueError(f"不支持的形状: {shape}")
        
        # 保存蒙版
        mask_img = Image.fromarray(mask.astype(np.uint8))
        mask_img.save(output_path)
        
        print(f"默认蒙版已创建: {output_path}")
        return True
        
    except Exception as e:
        warnings.warn(f"创建蒙版失败: {e}")
        return False


def add_wordcloud_annotations(image_path: str, 
                             top_words: List[str],
                             year: Optional[str] = None,
                             font_path: Optional[str] = None) -> bool:
    """
    为词云图片添加注释
    
    Args:
        image_path: 图片路径
        top_words: 前几个关键词
        year: 年份（可选）
        font_path: 字体路径
    
    Returns:
        bool: 是否成功
    """
    try:
        # 打开图片
        img = Image.open(image_path)
        draw = ImageDraw.Draw(img)
        
        # 设置字体
        try:
            if font_path and os.path.exists(font_path):
                font = ImageFont.truetype(font_path, 16)
            else:
                font = ImageFont.load_default()
        except:
            font = ImageFont.load_default()
        
        # 准备注释文本
        annotation_lines = []
        if year:
            annotation_lines.append(f"📅 {year}")
        
        if top_words:
            top_3 = top_words[:3]
            annotation_lines.append(f"🔥 热词: {' | '.join(top_3)}")
        
        # 绘制注释
        if annotation_lines:
            y_offset = img.height - 60
            for line in annotation_lines:
                # 添加文字阴影
                draw.text((11, y_offset + 1), line, fill="gray", font=font)
                # 添加文字
                draw.text((10, y_offset), line, fill="black", font=font)
                y_offset += 20
        
        # 保存图片
        img.save(image_path)
        return True
        
    except Exception as e:
        warnings.warn(f"添加注释失败: {e}")
        return False


def create_wordcloud_comparison(freq_data_dict: Dict[str, pd.DataFrame],
                               output_path: str,
                               font_path: Optional[str] = None,
                               top_n: int = 50) -> bool:
    """
    创建多年度词云对比图
    
    Args:
        freq_data_dict: 年度词频数据字典 {year: freq_dataframe}
        output_path: 输出路径
        font_path: 字体路径
        top_n: 每年使用的词数
    
    Returns:
        bool: 是否成功
    """
    if not freq_data_dict:
        return False
    
    try:
        years = sorted(freq_data_dict.keys())
        n_years = len(years)
        
        # 计算子图布局
        cols = min(3, n_years)
        rows = (n_years + cols - 1) // cols
        
        fig, axes = plt.subplots(rows, cols, figsize=(15, 5*rows))
        
        if n_years == 1:
            axes = [axes]
        elif rows == 1:
            axes = axes if isinstance(axes, list) else [axes]
        else:
            axes = axes.flatten()
        
        for i, year in enumerate(years):
            freq_data = freq_data_dict[year]
            
            if freq_data.empty:
                continue
            
            # 准备词频数据
            top_words = freq_data.head(top_n)
            # 确保数据类型正确
            word_freq = dict(zip(top_words['word'].astype(str), top_words['freq'].astype(int)))
            
            if not word_freq:
                continue
            
            # 创建词云
            wc_params = {
                'width': 400,
                'height': 300,
                'background_color': 'white',
                'max_words': top_n,
                'relative_scaling': 0.5,
                'random_state': 42
            }
            
            if font_path and os.path.exists(font_path):
                wc_params['font_path'] = font_path
            
            wordcloud = WordCloud(**wc_params)
            wordcloud.generate_from_frequencies(word_freq)
            
            # 显示词云
            axes[i].imshow(wordcloud, interpolation='bilinear')
            axes[i].set_title(f"{year} 年度词云", fontsize=14, fontweight='bold')
            axes[i].axis('off')
        
        # 隐藏多余的子图
        for i in range(n_years, len(axes)):
            axes[i].axis('off')
        
        plt.tight_layout()
        
        # 保存图片
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"词云对比图已保存: {output_path}")
        return True
        
    except Exception as e:
        warnings.warn(f"创建词云对比图失败: {e}")
        return False


if __name__ == "__main__":
    # 创建默认蒙版
    create_default_mask()