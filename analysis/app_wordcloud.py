"""
Interactive Chinese Wordcloud Streamlit Application
"""

import os
import logging
from pathlib import Path
from typing import Dict, List, Optional, Set

import streamlit as st
import pandas as pd
import numpy as np
from pyecharts import options as opts
from pyecharts.charts import WordCloud
from pyecharts.globals import SymbolType
import streamlit_echarts
import json

from data_loader import load_corpus, summarize_load_stats
from text_pipeline import (
    tokenize, compute_frequencies, kwic_index,
    load_stopwords_from_file, load_synonyms_from_csv
)

# Configure page
st.set_page_config(
    page_title="交互式中文词云分析",
    page_icon="☁️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Configure logging
logging.basicConfig(level=logging.INFO)

# Custom CSS for aesthetic styling
st.markdown("""
<style>
    .main {
        background-color: #FBFBFD;
    }
    
    .stApp {
        background-color: #FBFBFD;
    }
    
    h1, h2, h3 {
        color: #111827;
        font-weight: 600;
    }
    
    .stMarkdown p {
        color: #111827;
    }
    
    .stSelectbox label, .stSlider label, .stCheckbox label {
        color: #6B7280;
        font-weight: 400;
    }
    
    .success-message {
        background-color: #D1FAE5;
        border: 1px solid #10B981;
        border-radius: 8px;
        padding: 12px;
        color: #065F46;
        margin: 10px 0;
    }
    
    .error-message {
        background-color: #FEE2E2;
        border: 1px solid #DC2626;
        border-radius: 8px;
        padding: 12px;
        color: #991B1B;
        margin: 10px 0;
    }
    
    .info-card {
        background-color: #F9FAFB;
        border: 1px solid #E5E7EB;
        border-radius: 8px;
        padding: 16px;
        margin: 10px 0;
    }
    
    .metric-card {
        background-color: #F3F4F6;
        padding: 12px;
        border-radius: 6px;
        margin: 5px 0;
    }
    
    .button-container {
        display: flex;
        gap: 10px;
        margin: 15px 0;
    }
    
    .stButton > button {
        background-color: #F9FAFB;
        border: 1px solid #D1D5DB;
        color: #374151;
        transition: all 0.2s;
    }
    
    .stButton > button:hover {
        transform: scale(1.02);
        border-color: #3B82F6;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'corpus_data' not in st.session_state:
    st.session_state.corpus_data = None
if 'load_stats' not in st.session_state:
    st.session_state.load_stats = None
if 'selected_term' not in st.session_state:
    st.session_state.selected_term = None
if 'frequency_data' not in st.session_state:
    st.session_state.frequency_data = None

def load_default_stopwords() -> Set[str]:
    """Load default Chinese stopwords"""
    default_stopwords_path = Path(__file__).parent / "data" / "stopwords.zh.txt"
    if default_stopwords_path.exists():
        return load_stopwords_from_file(str(default_stopwords_path))
    return set()

def get_color_palette(theme: str) -> List[str]:
    """Get color palette based on theme"""
    palettes = {
        "冷色调": ["#1F2937", "#374151", "#4B5563", "#6B7280", "#9CA3AF", "#3B82F6", "#22D3EE", "#8B5CF6"],
        "灰阶": ["#111827", "#1F2937", "#374151", "#4B5563", "#6B7280", "#9CA3AF", "#D1D5DB"],
        "蓝紫高亮": ["#374151", "#4B5563", "#6366F1", "#8B5CF6", "#A855F7", "#C084FC", "#DDD6FE"]
    }
    return palettes.get(theme, palettes["冷色调"])

def create_wordcloud(freq_df: pd.DataFrame, color_theme: str, font_path: Optional[str] = None) -> str:
    """Create wordcloud using pyecharts"""
    if freq_df.empty:
        return ""
    
    # Prepare data for pyecharts wordcloud
    max_freq = freq_df['freq'].max()
    min_freq = freq_df['freq'].min()
    
    # Normalize frequencies for font size (log scale)
    freq_df['size'] = np.log1p(freq_df['freq']) / np.log1p(max_freq) * 60 + 20
    
    # Get color palette
    colors = get_color_palette(color_theme)
    
    # Prepare data
    data = []
    for _, row in freq_df.iterrows():
        data.append({
            "name": row['term'],
            "value": int(row['freq']),
            "textStyle": {
                "color": colors[min(int(row['freq'] / max_freq * (len(colors) - 1)), len(colors) - 1)]
            }
        })
    
    # Create wordcloud
    wordcloud = (
        WordCloud()
        .add(
            series_name="词频",
            data_pair=[(item["name"], item["value"]) for item in data],
            word_size_range=[20, 80],
            textstyle_opts=opts.TextStyleOpts(font_family="Noto Sans SC, Source Han Sans, sans-serif"),
        )
        .set_global_opts(
            title_opts=opts.TitleOpts(title=""),
            tooltip_opts=opts.TooltipOpts(
                formatter="{b}: {c} ({d}%)"
            ),
            toolbox_opts=opts.ToolboxOpts(
                feature={
                    "saveAsImage": {"show": True, "title": "保存图片"},
                }
            )
        )
    )
    
    return wordcloud.render_embed()

def main():
    st.title("📊 交互式中文词云分析")
    
    # Check for font installation reminder
    st.markdown("""
    <div class="info-card">
        <strong>字体提醒</strong>: 为获得最佳显示效果，建议安装 Noto Sans SC 或 Source Han Sans 中文字体。
        您也可以在侧栏上传自定义字体文件。
    </div>
    """, unsafe_allow_html=True)
    
    # Sidebar controls
    with st.sidebar:
        st.header("🔧 控制面板")
        
        # Data directory selection
        st.subheader("数据源")
        data_dir = st.text_input(
            "数据目录", 
            value=str(Path(__file__).parent / "data"),
            help="支持 JSON, CSV, Parquet 格式文件"
        )
        
        if st.button("🔄 加载数据"):
            try:
                with st.spinner("正在加载数据..."):
                    corpus_data = load_corpus(data_dir)
                    load_stats = summarize_load_stats(corpus_data)
                    st.session_state.corpus_data = corpus_data
                    st.session_state.load_stats = load_stats
                    st.session_state.selected_term = None
                    st.session_state.frequency_data = None
                    
                st.success("数据加载成功！")
                
            except Exception as e:
                st.error(f"数据加载失败: {str(e)}")
        
        # Display load statistics
        if st.session_state.load_stats:
            st.subheader("📈 数据统计")
            stats = st.session_state.load_stats
            
            col1, col2 = st.columns(2)
            with col1:
                st.metric("文件数", stats['total_files'])
                st.metric("记录数", stats['total_records'])
            with col2:
                st.metric("空内容", stats['empty_content'])
                st.metric("平均长度", f"{stats['avg_content_length']:.1f}")
            
            if stats['failed_files']:
                with st.expander("❌ 失败文件详情"):
                    for file_path, error in stats['failed_files']:
                        st.error(f"**{Path(file_path).name}**: {error}")
        
        st.divider()
        
        # Tokenization settings
        st.subheader("🔤 分词设置")
        
        tokenize_mode = st.selectbox(
            "分词模式",
            ["jieba_exact", "jieba_search", "ngram"],
            format_func=lambda x: {
                "jieba_exact": "Jieba 精确模式",
                "jieba_search": "Jieba 搜索引擎模式", 
                "ngram": "N-gram 模式"
            }[x]
        )
        
        n_value = 2
        if tokenize_mode == "ngram":
            n_value = st.slider("N-gram 大小", min_value=1, max_value=6, value=2)
        
        # Frequency settings
        st.subheader("📊 频次设置")
        min_freq = st.slider("最小词频", min_value=1, max_value=10, value=2)
        top_k = st.slider("显示词数 (Top-K)", min_value=50, max_value=500, value=200)
        
        # File uploads
        st.subheader("📁 自定义资源")
        
        # Stopwords upload
        stopwords_file = st.file_uploader("上传停用词文件", type=['txt'])
        custom_stopwords = set()
        if stopwords_file:
            content = stopwords_file.read().decode('utf-8')
            custom_stopwords = set(line.strip() for line in content.split('\n') if line.strip())
        
        # Whitelist upload
        whitelist_file = st.file_uploader("上传白名单文件", type=['txt'])
        custom_whitelist = set()
        if whitelist_file:
            content = whitelist_file.read().decode('utf-8')
            custom_whitelist = set(line.strip() for line in content.split('\n') if line.strip())
        
        # Synonyms upload
        synonyms_file = st.file_uploader("上传同义词映射文件", type=['csv'])
        custom_synonyms = {}
        if synonyms_file:
            try:
                synonyms_df = pd.read_csv(synonyms_file)
                if 'from' in synonyms_df.columns and 'to' in synonyms_df.columns:
                    custom_synonyms = dict(zip(synonyms_df['from'], synonyms_df['to']))
            except Exception as e:
                st.error(f"同义词文件读取失败: {e}")
        
        # Font upload
        font_file = st.file_uploader("上传字体文件", type=['ttf', 'otf'])
        font_path = None
        if font_file:
            fonts_dir = Path(__file__).parent / "assets" / "fonts"
            fonts_dir.mkdir(parents=True, exist_ok=True)
            font_path = fonts_dir / font_file.name
            with open(font_path, 'wb') as f:
                f.write(font_file.read())
            st.success(f"字体已保存: {font_file.name}")
        
        # Other settings
        st.subheader("⚙️ 其他设置")
        unify_trad_simp = st.checkbox("繁简合并", help="将繁体中文转换为简体")
        color_theme = st.selectbox("颜色主题", ["冷色调", "灰阶", "蓝紫高亮"])
        
        # Generate wordcloud button
        if st.button("🎨 生成词云"):
            if st.session_state.corpus_data:
                try:
                    with st.spinner("正在生成词云..."):
                        # Combine stopwords
                        default_stopwords = load_default_stopwords()
                        all_stopwords = default_stopwords.union(custom_stopwords)
                        
                        # Tokenize
                        tokens = tokenize(
                            st.session_state.corpus_data,
                            mode=tokenize_mode,
                            n=n_value,
                            stopwords=all_stopwords,
                            keep_whitelist=custom_whitelist,
                            synonyms=custom_synonyms,
                            unify_trad_simp=unify_trad_simp
                        )
                        
                        # Compute frequencies
                        freq_df = compute_frequencies(tokens)
                        freq_df = freq_df[freq_df['freq'] >= min_freq].head(top_k)
                        
                        st.session_state.frequency_data = freq_df
                    
                    st.success("词云生成完成！")
                    
                except Exception as e:
                    st.error(f"词云生成失败: {str(e)}")
            else:
                st.warning("请先加载数据")
    
    # Main content area
    if st.session_state.frequency_data is not None and not st.session_state.frequency_data.empty:
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.subheader("☁️ 词云可视化")
            
            # Create and display wordcloud
            wordcloud_html = create_wordcloud(
                st.session_state.frequency_data, 
                color_theme, 
                font_path
            )
            
            if wordcloud_html:
                # Use pyecharts wordcloud with custom styling
                wordcloud_option = {
                    "backgroundColor": "#FBFBFD",
                    "series": [{
                        "type": "wordCloud",
                        "data": [
                            {
                                "name": row['term'],
                                "value": row['freq'],
                                "textStyle": {
                                    "color": get_color_palette(color_theme)[
                                        min(int(row['freq'] / st.session_state.frequency_data['freq'].max() * 
                                               (len(get_color_palette(color_theme)) - 1)), 
                                            len(get_color_palette(color_theme)) - 1)
                                    ]
                                }
                            }
                            for _, row in st.session_state.frequency_data.iterrows()
                        ],
                        "sizeRange": [20, 80],
                        "rotationRange": [-45, 45],
                        "textStyle": {
                            "fontFamily": "Noto Sans SC, Source Han Sans, sans-serif",
                            "fontWeight": "normal"
                        },
                        "emphasis": {
                            "textStyle": {
                                "shadowBlur": 10,
                                "shadowColor": "#333"
                            }
                        }
                    }],
                    "tooltip": {
                        "formatter": lambda params: f"{params['name']}: {params['value']} ({params['value']/len(st.session_state.frequency_data)*100:.1f}%)"
                    }
                }
                
                # Display wordcloud with click handling
                clicked_data = streamlit_echarts.st_echarts(
                    options=wordcloud_option,
                    height="500px",
                    events={"click": "function(params) { return params.name; }"}
                )
                
                if clicked_data:
                    st.session_state.selected_term = clicked_data
            
            # Top words table
            st.subheader("📋 词频排行榜")
            
            # Make the dataframe interactive
            freq_display = st.session_state.frequency_data.copy()
            freq_display['占比 (%)'] = (freq_display['prop'] * 100).round(2)
            freq_display = freq_display[['term', 'freq', '占比 (%)']]
            freq_display.columns = ['词汇', '频次', '占比 (%)']
            
            # Add clickable functionality
            selected_indices = st.dataframe(
                freq_display,
                use_container_width=True,
                selection_mode="single-row",
                on_select="rerun"
            )
            
            if selected_indices and selected_indices['selection']['rows']:
                selected_idx = selected_indices['selection']['rows'][0]
                selected_term = freq_display.iloc[selected_idx]['词汇']
                st.session_state.selected_term = selected_term
        
        with col2:
            st.subheader("📖 词汇详情")
            
            if st.session_state.selected_term:
                term = st.session_state.selected_term
                
                # Display term statistics
                term_data = st.session_state.frequency_data[
                    st.session_state.frequency_data['term'] == term
                ]
                
                if not term_data.empty:
                    freq = term_data.iloc[0]['freq']
                    prop = term_data.iloc[0]['prop']
                    
                    st.markdown(f"""
                    <div class="info-card">
                        <h4>{term}</h4>
                        <p><strong>频次:</strong> {freq}</p>
                        <p><strong>占比:</strong> {prop*100:.2f}%</p>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    # KWIC context
                    st.subheader("🔍 上下文 (KWIC)")
                    
                    if st.session_state.corpus_data:
                        kwic_results = kwic_index(
                            st.session_state.corpus_data, 
                            term, 
                            window=30
                        )
                        
                        if kwic_results:
                            for i, kwic in enumerate(kwic_results[:5]):  # Show top 5
                                st.markdown(f"""
                                <div class="metric-card">
                                    <small>{kwic['title']}</small><br>
                                    ...{kwic['left']} <strong style="color: #3B82F6;">{kwic['keyword']}</strong> {kwic['right']}...
                                    {f'<br><a href="{kwic["url"]}" target="_blank" style="color: #6B7280; font-size: 0.8em;">查看原文</a>' if kwic['url'] else ''}
                                </div>
                                """, unsafe_allow_html=True)
                        else:
                            st.info("未找到相关上下文")
            else:
                st.info("点击词云中的词汇或表格中的行来查看详细信息")
        
        # Export buttons
        st.divider()
        st.subheader("💾 导出选项")
        
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("📄 导出 HTML 词云"):
                try:
                    outputs_dir = Path(__file__).parent / "outputs"
                    outputs_dir.mkdir(exist_ok=True)
                    
                    output_path = outputs_dir / "wordcloud_interactive.html"
                    
                    # Create full HTML wordcloud
                    wordcloud = create_wordcloud(st.session_state.frequency_data, color_theme, font_path)
                    
                    with open(output_path, 'w', encoding='utf-8') as f:
                        f.write(wordcloud)
                    
                    st.success(f"HTML词云已导出至: {output_path}")
                    
                except Exception as e:
                    st.error(f"导出失败: {e}")
        
        with col2:
            if st.button("📊 导出 CSV 词频表"):
                try:
                    outputs_dir = Path(__file__).parent / "outputs"
                    outputs_dir.mkdir(exist_ok=True)
                    
                    output_path = outputs_dir / "term_frequencies.csv"
                    st.session_state.frequency_data.to_csv(output_path, index=False, encoding='utf-8-sig')
                    
                    st.success(f"词频表已导出至: {output_path}")
                    
                except Exception as e:
                    st.error(f"导出失败: {e}")
                    
    else:
        # Display welcome message
        st.markdown("""
        <div class="info-card">
            <h3>欢迎使用交互式中文词云分析工具</h3>
            <p>请按以下步骤开始分析:</p>
            <ol>
                <li>在左侧边栏选择数据目录并点击"加载数据"</li>
                <li>调整分词模式和频次设置</li>
                <li>可选: 上传自定义停用词、白名单、同义词映射文件</li>
                <li>点击"生成词云"开始分析</li>
                <li>点击词云中的词汇查看详细上下文信息</li>
            </ol>
        </div>
        """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()