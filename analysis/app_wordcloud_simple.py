"""
Simplified Interactive Chinese Wordcloud Streamlit Application
Works with minimal dependencies - demonstrates the complete functionality
"""

import os
import logging
from pathlib import Path
from typing import Dict, List, Optional, Set
import json
import streamlit as st

from data_loader_simple import (
    load_corpus_simple, summarize_load_stats_simple, 
    simple_chinese_tokenize, compute_frequencies_simple
)

# Configure page
st.set_page_config(
    page_title="交互式中文词云分析 (简化版)",
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
    
    .wordcloud-container {
        background-color: #FBFBFD;
        padding: 20px;
        border-radius: 8px;
        text-align: center;
        min-height: 400px;
        border: 1px solid #E5E7EB;
    }
    
    .word-item {
        display: inline-block;
        margin: 5px;
        padding: 3px 8px;
        background-color: #3B82F6;
        color: white;
        border-radius: 4px;
        cursor: pointer;
        transition: all 0.2s;
    }
    
    .word-item:hover {
        background-color: #1D4ED8;
        transform: scale(1.1);
    }
    
    .frequency-table {
        background-color: #F9FAFB;
        border-radius: 6px;
        padding: 10px;
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

def create_simple_wordcloud_html(freq_data: List[Dict], selected_term: str = None) -> str:
    """Create a simple HTML wordcloud visualization"""
    if not freq_data:
        return "<div class='wordcloud-container'><p>没有数据可显示</p></div>"
    
    max_freq = max(item['freq'] for item in freq_data)
    min_freq = min(item['freq'] for item in freq_data)
    
    words_html = []
    colors = ["#1F2937", "#374151", "#4B5563", "#3B82F6", "#22D3EE", "#8B5CF6"]
    
    for item in freq_data[:50]:  # Show top 50 words
        term = item['term']
        freq = item['freq']
        
        # Calculate font size (12-36px)
        if max_freq == min_freq:
            font_size = 20
        else:
            font_size = 12 + (freq - min_freq) / (max_freq - min_freq) * 24
        
        # Calculate color
        color_idx = min(int((freq - min_freq) / (max_freq - min_freq) * len(colors)), len(colors) - 1)
        color = colors[color_idx]
        
        # Highlight selected term
        style_extra = ""
        if term == selected_term:
            style_extra = "background-color: #EF4444; color: white;"
        
        word_html = f"""
        <span class="word-item" 
              style="font-size: {font_size}px; color: {color}; {style_extra}"
              onclick="selectWord('{term}')">
            {term} ({freq})
        </span>
        """
        words_html.append(word_html)
    
    javascript = """
    <script>
    function selectWord(term) {
        // This would normally communicate back to Streamlit
        alert('选中词汇: ' + term + '\\n(在完整版本中会显示详细信息)');
    }
    </script>
    """
    
    html = f"""
    <div class="wordcloud-container">
        <h4 style="color: #111827; margin-bottom: 20px;">词云可视化</h4>
        {''.join(words_html)}
        {javascript}
    </div>
    """
    
    return html

def find_kwic_contexts(records: List[Dict], term: str, window: int = 20) -> List[Dict]:
    """Find keyword in context"""
    contexts = []
    
    for record in records:
        content = record.get('content', '')
        title = record.get('title', '未知标题')
        url = record.get('url', '')
        
        if not content or term not in content:
            continue
        
        # Find all occurrences
        start = 0
        while True:
            pos = content.find(term, start)
            if pos == -1:
                break
            
            # Extract context
            left_start = max(0, pos - window)
            right_end = min(len(content), pos + len(term) + window)
            
            left_context = content[left_start:pos]
            right_context = content[pos + len(term):right_end]
            
            contexts.append({
                'left': left_context,
                'keyword': term,
                'right': right_context,
                'title': title,
                'url': url
            })
            
            start = pos + 1
    
    return contexts

def main():
    st.title("📊 交互式中文词云分析 (简化版)")
    
    # Information about the simplified version
    st.markdown("""
    <div class="info-card">
        <strong>简化版说明</strong>: 这是使用标准库的简化版本。完整版本需要安装额外的依赖包（pandas, pyecharts等）来获得更丰富的功能。
        <br><br>
        <strong>字体提醒</strong>: 为获得最佳显示效果，建议安装 Noto Sans SC 或 Source Han Sans 中文字体。
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
            help="支持 JSON, CSV 格式文件"
        )
        
        if st.button("🔄 加载数据"):
            try:
                with st.spinner("正在加载数据..."):
                    corpus_data = load_corpus_simple(data_dir)
                    load_stats = summarize_load_stats_simple(corpus_data)
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
        
        # Frequency settings
        st.subheader("📊 分析设置")
        min_freq = st.slider("最小词频", min_value=1, max_value=10, value=2)
        top_k = st.slider("显示词数 (Top-K)", min_value=20, max_value=200, value=50)
        
        # Simple stopwords input
        st.subheader("🔤 停用词设置")
        default_stopwords = "的,是,在,和,与,或,但,而,了,着,过,要,会,能,可,将,已,被,把,给,从,向,到,为,由,以,及"
        stopwords_text = st.text_area(
            "停用词 (用逗号分隔)", 
            value=default_stopwords,
            help="输入要过滤的停用词，用逗号分隔"
        )
        
        # Generate wordcloud button
        if st.button("🎨 生成词云"):
            if st.session_state.corpus_data:
                try:
                    with st.spinner("正在生成词云..."):
                        # Get all content
                        all_content = " ".join([
                            record['content'] for record in st.session_state.corpus_data 
                            if record['content']
                        ])
                        
                        # Tokenize
                        tokens = simple_chinese_tokenize(all_content)
                        
                        # Filter stopwords
                        stopwords = set(word.strip() for word in stopwords_text.split(',') if word.strip())
                        filtered_tokens = [token for token in tokens if token not in stopwords and len(token) > 1]
                        
                        # Compute frequencies
                        freq_data = compute_frequencies_simple(filtered_tokens)
                        freq_data = [item for item in freq_data if item['freq'] >= min_freq][:top_k]
                        
                        st.session_state.frequency_data = freq_data
                    
                    st.success("词云生成完成！")
                    
                except Exception as e:
                    st.error(f"词云生成失败: {str(e)}")
            else:
                st.warning("请先加载数据")
    
    # Main content area
    if st.session_state.frequency_data:
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.subheader("☁️ 词云可视化")
            
            # Display simple wordcloud
            wordcloud_html = create_simple_wordcloud_html(
                st.session_state.frequency_data,
                st.session_state.selected_term
            )
            st.markdown(wordcloud_html, unsafe_allow_html=True)
            
            # Word selection
            st.subheader("🔍 选择词汇查看详情")
            if st.session_state.frequency_data:
                term_options = [item['term'] for item in st.session_state.frequency_data]
                selected_term = st.selectbox(
                    "选择词汇", 
                    options=[""] + term_options,
                    help="选择一个词汇查看详细信息"
                )
                if selected_term:
                    st.session_state.selected_term = selected_term
            
            # Top words table
            st.subheader("📋 词频排行榜")
            
            # Create a simple table
            table_data = []
            for i, item in enumerate(st.session_state.frequency_data[:20]):
                table_data.append({
                    "排名": i + 1,
                    "词汇": item['term'],
                    "频次": item['freq'],
                    "占比 (%)": f"{item['prop'] * 100:.2f}%"
                })
            
            if table_data:
                st.table(table_data)
        
        with col2:
            st.subheader("📖 词汇详情")
            
            if st.session_state.selected_term:
                term = st.session_state.selected_term
                
                # Find term in frequency data
                term_data = None
                for item in st.session_state.frequency_data:
                    if item['term'] == term:
                        term_data = item
                        break
                
                if term_data:
                    st.markdown(f"""
                    <div class="info-card">
                        <h4>{term}</h4>
                        <p><strong>频次:</strong> {term_data['freq']}</p>
                        <p><strong>占比:</strong> {term_data['prop']*100:.2f}%</p>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    # KWIC context
                    st.subheader("🔍 上下文 (KWIC)")
                    
                    if st.session_state.corpus_data:
                        kwic_results = find_kwic_contexts(
                            st.session_state.corpus_data, 
                            term, 
                            window=20
                        )
                        
                        if kwic_results:
                            for i, kwic in enumerate(kwic_results[:5]):  # Show top 5
                                st.markdown(f"""
                                <div class="metric-card">
                                    <small><strong>{kwic['title']}</strong></small><br>
                                    ...{kwic['left']} <strong style="color: #3B82F6;">{kwic['keyword']}</strong> {kwic['right']}...
                                    {f'<br><a href="{kwic["url"]}" target="_blank" style="color: #6B7280; font-size: 0.8em;">查看原文</a>' if kwic['url'] else ''}
                                </div>
                                """, unsafe_allow_html=True)
                        else:
                            st.info("未找到相关上下文")
                else:
                    st.error("未找到选中词汇的数据")
            else:
                st.info("在左侧选择一个词汇来查看详细信息")
        
        # Export section
        st.divider()
        st.subheader("💾 导出选项")
        
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("📄 导出词频数据 (JSON)"):
                try:
                    outputs_dir = Path(__file__).parent / "outputs"
                    outputs_dir.mkdir(exist_ok=True)
                    
                    output_path = outputs_dir / "word_frequencies.json"
                    
                    with open(output_path, 'w', encoding='utf-8') as f:
                        json.dump(st.session_state.frequency_data, f, ensure_ascii=False, indent=2)
                    
                    st.success(f"词频数据已导出至: {output_path}")
                    
                except Exception as e:
                    st.error(f"导出失败: {e}")
        
        with col2:
            if st.button("🔍 导出KWIC数据"):
                if st.session_state.selected_term:
                    try:
                        outputs_dir = Path(__file__).parent / "outputs"
                        outputs_dir.mkdir(exist_ok=True)
                        
                        kwic_results = find_kwic_contexts(
                            st.session_state.corpus_data, 
                            st.session_state.selected_term, 
                            window=30
                        )
                        
                        output_path = outputs_dir / f"kwic_{st.session_state.selected_term}.json"
                        
                        with open(output_path, 'w', encoding='utf-8') as f:
                            json.dump(kwic_results, f, ensure_ascii=False, indent=2)
                        
                        st.success(f"KWIC数据已导出至: {output_path}")
                        
                    except Exception as e:
                        st.error(f"导出失败: {e}")
                else:
                    st.warning("请先选择一个词汇")
                    
    else:
        # Display welcome message
        st.markdown("""
        <div class="info-card">
            <h3>欢迎使用交互式中文词云分析工具 (简化版)</h3>
            <p>请按以下步骤开始分析:</p>
            <ol>
                <li>在左侧边栏选择数据目录并点击"加载数据"</li>
                <li>调整最小词频和显示词数设置</li>
                <li>可选: 自定义停用词列表</li>
                <li>点击"生成词云"开始分析</li>
                <li>在词云下方选择词汇查看详细上下文信息</li>
            </ol>
            <br>
            <p><strong>支持的数据格式:</strong></p>
            <ul>
                <li>JSON 文件: 包含 title, content 等字段的对象数组</li>
                <li>CSV 文件: 包含标题、内容等列的表格数据</li>
            </ul>
            <br>
            <p><strong>当前已加载的示例数据:</strong> analysis/data/sample_articles.json</p>
        </div>
        """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()