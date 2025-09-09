#!/usr/bin/env python3
"""
演示脚本：内容发现与元数据提取
基于TOC.md中的第一阶段指令实现
"""

import os
import re
from pathlib import Path
import json
from collections import defaultdict

def scan_directory_structure(root_dir):
    """扫描目录结构，发现所有文章"""
    root = Path(root_dir)
    articles = []
    
    # 扫描年份目录
    for year_dir in sorted(root.glob("[0-9][0-9][0-9][0-9]")):
        year = year_dir.name
        print(f"扫描年份: {year}")
        
        # 扫描文章目录
        for article_dir in year_dir.iterdir():
            if not article_dir.is_dir():
                continue
                
            # 检查目录名格式: YYYY-MM-DD_标题
            match = re.match(r'(\d{4}-\d{2}-\d{2})_(.+)', article_dir.name)
            if not match:
                continue
                
            date, title = match.groups()
            
            # 检查文件类型
            md_file = article_dir / f"{article_dir.name}.md"
            html_file = article_dir / f"{article_dir.name}.html"
            images_dir = article_dir / "images"
            meta_file = article_dir / "meta.json"
            
            # 统计字数（如果MD文件存在）
            word_count = 0
            image_count = 0
            
            if md_file.exists():
                with open(md_file, 'r', encoding='utf-8') as f:
                    content = f.read()
                    # 简单字数统计（去除markdown语法）
                    text = re.sub(r'[#*`\[\]()!]', '', content)
                    text = re.sub(r'https?://\S+', '', text)  # 去除链接
                    word_count = len(re.findall(r'[\u4e00-\u9fff]', text))  # 中文字符
                    
                    # 统计图片数量
                    image_count += len(re.findall(r'!\[.*?\]\(', content))
            
            # 统计本地图片文件
            if images_dir.exists():
                image_count += len([f for f in images_dir.iterdir() if f.suffix.lower() in ['.jpg', '.jpeg', '.png', '.gif']])
            
            articles.append({
                'year': year,
                'date': date,
                'title': title,
                'word_count': word_count,
                'image_count': image_count,
                'has_md': md_file.exists(),
                'has_html': html_file.exists(),
                'has_meta': meta_file.exists(),
                'has_images': images_dir.exists(),
                'directory': str(article_dir.relative_to(root))
            })
    
    return articles

def generate_statistics(articles):
    """生成统计数据"""
    stats = {
        'total_articles': len(articles),
        'total_words': sum(a['word_count'] for a in articles),
        'total_images': sum(a['image_count'] for a in articles),
        'by_year': defaultdict(lambda: {'articles': 0, 'words': 0, 'images': 0})
    }
    
    for article in articles:
        year_stats = stats['by_year'][article['year']]
        year_stats['articles'] += 1
        year_stats['words'] += article['word_count']
        year_stats['images'] += article['image_count']
    
    return stats

def main():
    # 设置目标目录
    target_dir = "Wechat-Backup/文不加点的张衔瑜"
    
    if not os.path.exists(target_dir):
        print(f"目录不存在: {target_dir}")
        return
    
    print("=== 内容发现与元数据提取演示 ===")
    print(f"扫描目录: {target_dir}")
    print()
    
    # 执行扫描
    articles = scan_directory_structure(target_dir)
    stats = generate_statistics(articles)
    
    # 输出统计报告
    print("=== 整体统计 ===")
    print(f"总文章数: {stats['total_articles']}篇")
    print(f"总字数: {stats['total_words']:,}字")
    print(f"总图片数: {stats['total_images']}图")
    print()
    
    print("=== 年度统计 ===")
    for year in sorted(stats['by_year'].keys()):
        year_stats = stats['by_year'][year]
        print(f"{year}年: {year_stats['articles']}篇 | {year_stats['words']:,}字 | {year_stats['images']}图")
    
    print()
    print("=== 最新5篇文章示例 ===")
    latest_articles = sorted(articles, key=lambda x: x['date'], reverse=True)[:5]
    for i, article in enumerate(latest_articles, 1):
        print(f"{i}. {article['date']} - {article['title']}")
        print(f"   字数: {article['word_count']:,} | 图片: {article['image_count']}")
        print(f"   路径: {article['directory']}")
        print()
    
    # 保存详细报告
    report_file = "script/discovery_report.json"
    with open(report_file, 'w', encoding='utf-8') as f:
        json.dump({
            'statistics': dict(stats),
            'articles': articles[:10]  # 保存前10篇作为示例
        }, f, ensure_ascii=False, indent=2)
    
    print(f"详细报告已保存到: {report_file}")

if __name__ == "__main__":
    main()