"""
语料装载模块 - 读取 WeChat 备份目录中的文章内容
"""

import os
import json
import re
from pathlib import Path
from dataclasses import dataclass
from typing import List, Optional
import warnings
from bs4 import BeautifulSoup
from datetime import datetime


@dataclass
class Article:
    """文章数据结构"""
    date: str
    year: str
    title: str
    html_path: str
    md_path: str
    meta_path: str
    content: Optional[str] = None


def scan_articles(root_dir: str, start_date: Optional[str] = None, 
                  end_date: Optional[str] = None, years: Optional[List[str]] = None) -> List[Article]:
    """
    扫描根目录下的所有文章
    
    Args:
        root_dir: 根目录路径，如 'Wechat-Backup/文不加点的张衔瑜'
        start_date: 开始日期，格式 'YYYY-MM-DD'
        end_date: 结束日期，格式 'YYYY-MM-DD'
        years: 年份白名单，如 ['2021', '2022', '2023']
    
    Returns:
        List[Article]: 文章列表
    """
    articles = []
    root_path = Path(root_dir)
    
    if not root_path.exists():
        warnings.warn(f"根目录不存在: {root_dir}")
        return articles
    
    # 扫描年份目录
    for year_dir in root_path.iterdir():
        if not year_dir.is_dir() or not year_dir.name.isdigit():
            continue
            
        year = year_dir.name
        
        # 年份过滤
        if years and year not in years:
            continue
            
        # 扫描文章目录
        for article_dir in year_dir.iterdir():
            if not article_dir.is_dir():
                continue
                
            try:
                # 解析文章目录名：YYYY-MM-DD_标题
                dir_name = article_dir.name
                if not re.match(r'\d{4}-\d{2}-\d{2}_', dir_name):
                    continue
                    
                date_part = dir_name.split('_')[0]
                title = '_'.join(dir_name.split('_')[1:])
                
                # 日期过滤
                if start_date and date_part < start_date:
                    continue
                if end_date and date_part > end_date:
                    continue
                
                # 文件路径
                html_path = article_dir / f"{dir_name}.html"
                md_path = article_dir / f"{dir_name}.md"
                meta_path = article_dir / "meta.json"
                
                # 验证必要文件存在
                if not meta_path.exists():
                    warnings.warn(f"meta.json 不存在: {article_dir}")
                    continue
                
                # 创建文章对象
                article = Article(
                    date=date_part,
                    year=year,
                    title=title,
                    html_path=str(html_path) if html_path.exists() else "",
                    md_path=str(md_path) if md_path.exists() else "",
                    meta_path=str(meta_path)
                )
                
                articles.append(article)
                
            except Exception as e:
                warnings.warn(f"处理文章目录失败: {article_dir}, 错误: {e}")
                continue
    
    # 按日期排序
    articles.sort(key=lambda x: x.date)
    
    print(f"扫描完成，共找到 {len(articles)} 篇文章")
    return articles


def read_text(article: Article) -> str:
    """
    读取文章正文内容
    
    Args:
        article: 文章对象
    
    Returns:
        str: 文章正文
    """
    content = ""
    
    try:
        # 优先读取 Markdown 文件
        if article.md_path and os.path.exists(article.md_path):
            with open(article.md_path, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read()
                content = _clean_markdown(content)
        # 否则从 HTML 提取
        elif article.html_path and os.path.exists(article.html_path):
            with open(article.html_path, 'r', encoding='utf-8', errors='ignore') as f:
                html_content = f.read()
                content = _extract_text_from_html(html_content)
        else:
            warnings.warn(f"文章文件不存在: {article.title}")
            
    except Exception as e:
        warnings.warn(f"读取文章失败: {article.title}, 错误: {e}")
    
    return content


def _clean_markdown(content: str) -> str:
    """清理 Markdown 内容"""
    # 移除图片链接
    content = re.sub(r'!\[.*?\]\(.*?\)', '', content)
    # 移除链接
    content = re.sub(r'\[([^\]]+)\]\([^)]+\)', r'\1', content)
    # 移除代码块
    content = re.sub(r'```.*?```', '', content, flags=re.DOTALL)
    content = re.sub(r'`([^`]+)`', r'\1', content)
    # 移除 Markdown 标记
    content = re.sub(r'#+\s*', '', content)
    content = re.sub(r'\*\*([^*]+)\*\*', r'\1', content)
    content = re.sub(r'\*([^*]+)\*', r'\1', content)
    # 清理多余空白
    content = re.sub(r'\n\s*\n', '\n', content)
    content = content.strip()
    
    return content


def _extract_text_from_html(html_content: str) -> str:
    """从 HTML 中提取正文"""
    try:
        soup = BeautifulSoup(html_content, 'lxml')
        
        # 移除脚本和样式
        for script in soup(["script", "style"]):
            script.decompose()
        
        # 查找主要内容区域
        content_div = soup.find('div', {'id': 'js_content'}) or \
                     soup.find('div', class_='rich_media_content') or \
                     soup.find('body')
        
        if content_div:
            text = content_div.get_text()
        else:
            text = soup.get_text()
        
        # 清理文本
        text = re.sub(r'\s+', ' ', text)
        text = text.strip()
        
        return text
        
    except Exception as e:
        warnings.warn(f"HTML 解析失败: {e}")
        return ""


def load_meta(article: Article) -> dict:
    """
    加载文章元数据
    
    Args:
        article: 文章对象
    
    Returns:
        dict: 元数据字典
    """
    try:
        with open(article.meta_path, 'r', encoding='utf-8', errors='ignore') as f:
            meta = json.load(f)
            return meta
    except Exception as e:
        warnings.warn(f"读取元数据失败: {article.title}, 错误: {e}")
        return {}


def get_corpus_stats(articles: List[Article]) -> dict:
    """
    获取语料库统计信息
    
    Args:
        articles: 文章列表
    
    Returns:
        dict: 统计信息
    """
    if not articles:
        return {}
    
    total_articles = len(articles)
    years = sorted(list(set(article.year for article in articles)))
    date_range = (articles[0].date, articles[-1].date)
    
    # 按年统计
    year_stats = {}
    for year in years:
        year_articles = [a for a in articles if a.year == year]
        year_stats[year] = len(year_articles)
    
    return {
        'total_articles': total_articles,
        'years': years,
        'date_range': date_range,
        'articles_by_year': year_stats
    }