#!/usr/bin/env python3
"""
I/O utilities for corpus loading and year-based splitting
"""

import os
import re
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional


@dataclass
class Article:
    """Article data structure"""
    date: str
    year: str
    title: str
    html_path: str
    md_path: str
    meta_path: str
    content: Optional[str] = None


def load_corpus(root_dir: str, start_date: Optional[str] = None,
                end_date: Optional[str] = None, years: Optional[List[str]] = None) -> List[Article]:
    """
    Load corpus from WeChat backup directory
    
    Args:
        root_dir: Root directory path, e.g. 'Wechat-Backup/文不加点的张衔瑜'
        start_date: Start date in 'YYYY-MM-DD' format
        end_date: End date in 'YYYY-MM-DD' format
        years: Year whitelist, e.g. ['2021', '2022', '2023']
    
    Returns:
        List[Article]: List of articles
    """
    articles = []
    root_path = Path(root_dir)

    if not root_path.exists():
        warnings.warn(f"Root directory does not exist: {root_dir}")
        return articles

    # Scan year directories
    for year_dir in root_path.iterdir():
        if not year_dir.is_dir() or not year_dir.name.isdigit():
            continue

        year = year_dir.name

        # Filter by years if specified
        if years and year not in years:
            continue

        # Scan articles in year directory
        for article_dir in year_dir.iterdir():
            if not article_dir.is_dir():
                continue

            # Extract date from directory name
            match = re.match(r'(\d{4}-\d{2}-\d{2})', article_dir.name)
            if not match:
                continue

            date = match.group(1)

            # Filter by date range if specified
            if start_date and date < start_date:
                continue
            if end_date and date > end_date:
                continue

            title = article_dir.name.replace(f"{date}_", "")

            # Look for content files
            md_file = article_dir / f"{article_dir.name}.md"
            html_file = article_dir / f"{article_dir.name}.html"
            meta_file = article_dir / "meta.json"

            if md_file.exists():
                article = Article(
                    date=date,
                    year=year,
                    title=title,
                    html_path=str(html_file) if html_file.exists() else "",
                    md_path=str(md_file),
                    meta_path=str(meta_file) if meta_file.exists() else ""
                )
                articles.append(article)

    # Sort by date
    articles.sort(key=lambda x: x.date)
    return articles


def read_text(article: Article) -> str:
    """
    Read text content from article
    
    Args:
        article: Article object
        
    Returns:
        str: Article text content
    """
    if not article.md_path or not os.path.exists(article.md_path):
        return ""

    try:
        with open(article.md_path, encoding='utf-8') as f:
            content = f.read()

        # Simple markdown cleanup - remove images, links, headers
        content = re.sub(r'!\[.*?\]\(.*?\)', '', content)  # Remove images
        content = re.sub(r'\[.*?\]\(.*?\)', '', content)   # Remove links
        content = re.sub(r'^#+\s*', '', content, flags=re.MULTILINE)  # Remove headers
        content = re.sub(r'\n+', '\n', content)  # Normalize newlines

        return content.strip()
    except Exception as e:
        warnings.warn(f"Failed to read article {article.md_path}: {e}")
        return ""


def get_corpus_stats(articles: List[Article]) -> Dict[str, Any]:
    """
    Get basic corpus statistics
    
    Args:
        articles: List of articles
        
    Returns:
        Dict: Corpus statistics
    """
    if not articles:
        return {}

    years = sorted(set(article.year for article in articles))
    articles_by_year = {}

    for article in articles:
        year = article.year
        if year not in articles_by_year:
            articles_by_year[year] = 0
        articles_by_year[year] += 1

    return {
        'total_articles': len(articles),
        'years': years,
        'year_range': f"{min(years)}-{max(years)}" if len(years) > 1 else years[0],
        'articles_by_year': articles_by_year,
        'date_range': f"{articles[0].date} to {articles[-1].date}"
    }


def split_by_year(articles: List[Article]) -> Dict[str, List[Article]]:
    """
    Split articles by year
    
    Args:
        articles: List of articles
        
    Returns:
        Dict[str, List[Article]]: Articles grouped by year
    """
    by_year = {}
    for article in articles:
        year = article.year
        if year not in by_year:
            by_year[year] = []
        by_year[year].append(article)

    return by_year
