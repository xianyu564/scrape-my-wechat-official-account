#!/usr/bin/env python3
"""
Style SFT Data Builder
从 Markdown + meta.json 生成 SFT JSONL（指令微调数据）

用途：将微信公众号的文章内容转换为适合进行风格微调的训练数据
- 扫描 wechat-backup/**/目录下的 .md 文件和 meta.json
- 按 300-800 tokens 切块，生成多种指令模板
- 输出训练集和验证集 JSONL 文件
"""

import os
import re
import json
import hashlib
import argparse
from pathlib import Path
from typing import List, Dict, Tuple, Set
from collections import defaultdict
import random

def clean_markdown_content(content: str) -> str:
    """清理 Markdown 内容，去除不必要的格式"""
    # 去除图片引用
    content = re.sub(r'!\[.*?\]\(.*?\)', '', content)
    # 去除超链接，保留文本
    content = re.sub(r'\[([^\]]+)\]\([^\)]+\)', r'\1', content)
    # 去除标题符号
    content = re.sub(r'^#+\s*', '', content, flags=re.MULTILINE)
    # 去除多余空行
    content = re.sub(r'\n\s*\n\s*\n', '\n\n', content)
    # 去除行首数字编号
    content = re.sub(r'^\d+\.\s*', '', content, flags=re.MULTILINE)
    return content.strip()

def split_into_chunks(text: str, min_tokens: int = 300, max_tokens: int = 800) -> List[str]:
    """将文本切分为适当大小的块"""
    # 简单的字符数估算（中文约1字符=1token，英文约4字符=1token）
    min_chars = min_tokens
    max_chars = max_tokens
    
    chunks = []
    paragraphs = text.split('\n\n')
    current_chunk = ""
    
    for para in paragraphs:
        para = para.strip()
        if not para:
            continue
            
        # 如果当前段落太长，需要进一步切分
        if len(para) > max_chars:
            # 按句号切分
            sentences = re.split(r'[。！？]', para)
            for sentence in sentences:
                sentence = sentence.strip()
                if not sentence:
                    continue
                sentence += '。'  # 恢复句号
                
                if len(current_chunk + sentence) > max_chars:
                    if len(current_chunk) >= min_chars:
                        chunks.append(current_chunk.strip())
                    current_chunk = sentence
                else:
                    current_chunk += sentence
        else:
            if len(current_chunk + para) > max_chars:
                if len(current_chunk) >= min_chars:
                    chunks.append(current_chunk.strip())
                current_chunk = para
            else:
                if current_chunk:
                    current_chunk += '\n\n' + para
                else:
                    current_chunk = para
    
    # 添加最后一个块
    if current_chunk and len(current_chunk) >= min_chars:
        chunks.append(current_chunk.strip())
    
    return [chunk for chunk in chunks if len(chunk) >= min_chars]

def generate_training_templates(chunk: str, meta_info: Dict) -> List[Dict]:
    """为文本块生成多种训练模板"""
    templates = []
    
    # 模板1：改写指令
    templates.append({
        "system": "你是该公众号作者，保持其叙述节奏与转折。",
        "input": f"用我的口吻改写下面这段材料：{chunk}",
        "output": chunk,
        "meta": meta_info
    })
    
    # 模板2：续写指令（取前半段，要求续写）
    if len(chunk) > 200:
        split_point = len(chunk) // 2
        # 找到合适的切分点（句号后）
        for i in range(split_point, min(split_point + 100, len(chunk))):
            if chunk[i] in '。！？':
                split_point = i + 1
                break
        
        prefix = chunk[:split_point].strip()
        suffix = chunk[split_point:].strip()
        
        if prefix and suffix:
            templates.append({
                "system": "你是该公众号作者，保持其叙述节奏与转折。",
                "input": f"请用我的风格续写下面的文字：{prefix}",
                "output": chunk,
                "meta": meta_info
            })
    
    # 模板3：总结再展开
    if len(chunk) > 400:
        templates.append({
            "system": "你是该公众号作者，保持其叙述节奏与转折。",
            "input": f"用我的写作风格，围绕以下内容写一段文字：{chunk[:100]}...",
            "output": chunk,
            "meta": meta_info
        })
    
    return templates

def calculate_text_hash(text: str) -> str:
    """计算文本的简单哈希，用于去重"""
    return hashlib.md5(text.encode('utf-8')).hexdigest()

def is_similar_content(text1: str, text2: str, threshold: float = 0.8) -> bool:
    """简单的相似度检测（基于重叠词汇）"""
    words1 = set(text1)
    words2 = set(text2)
    
    if not words1 or not words2:
        return False
    
    intersection = len(words1.intersection(words2))
    union = len(words1.union(words2))
    
    return intersection / union > threshold

def load_wechat_backup_data(backup_dir: Path) -> List[Tuple[str, Dict]]:
    """加载微信备份数据"""
    articles = []
    
    for md_file in backup_dir.rglob("*.md"):
        # 跳过索引文件
        if md_file.name in ['目录.md', '合集.md', 'action_suggestion.md']:
            continue
            
        try:
            # 读取 Markdown 内容
            with open(md_file, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # 查找对应的 meta.json
            meta_file = md_file.parent / 'meta.json'
            meta_info = {}
            
            if meta_file.exists():
                with open(meta_file, 'r', encoding='utf-8') as f:
                    meta_info = json.load(f)
            
            # 添加路径信息用于主题分析
            meta_info['file_path'] = str(md_file.relative_to(backup_dir))
            meta_info['year'] = meta_info.get('year', 'unknown')
            
            # 清理内容
            cleaned_content = clean_markdown_content(content)
            
            if len(cleaned_content) > 100:  # 过滤太短的内容
                articles.append((cleaned_content, meta_info))
                
        except Exception as e:
            print(f"处理文件 {md_file} 时出错: {e}")
            continue
    
    return articles

def build_sft_dataset(backup_dir: Path, output_dir: Path, train_ratio: float = 0.9):
    """构建 SFT 数据集"""
    print(f"扫描备份目录: {backup_dir}")
    
    # 加载数据
    articles = load_wechat_backup_data(backup_dir)
    print(f"找到 {len(articles)} 篇文章")
    
    if not articles:
        print("未找到任何文章，请检查备份目录路径")
        return
    
    # 生成训练样本
    all_samples = []
    seen_hashes = set()
    
    for content, meta_info in articles:
        chunks = split_into_chunks(content)
        print(f"文章 {meta_info.get('title', '无标题')} 切分为 {len(chunks)} 个块")
        
        for chunk in chunks:
            # 去重检查
            chunk_hash = calculate_text_hash(chunk)
            if chunk_hash in seen_hashes:
                continue
            seen_hashes.add(chunk_hash)
            
            # 生成训练模板
            templates = generate_training_templates(chunk, meta_info)
            all_samples.extend(templates)
    
    print(f"生成 {len(all_samples)} 个训练样本")
    
    # 打乱并分割数据集
    random.shuffle(all_samples)
    split_point = int(len(all_samples) * train_ratio)
    
    train_samples = all_samples[:split_point]
    val_samples = all_samples[split_point:]
    
    # 保存数据集
    output_dir.mkdir(parents=True, exist_ok=True)
    
    train_file = output_dir / 'sft_train.jsonl'
    val_file = output_dir / 'sft_val.jsonl'
    
    # 写入训练集
    with open(train_file, 'w', encoding='utf-8') as f:
        for sample in train_samples:
            # 移除 meta 信息，只保留训练需要的字段
            training_sample = {
                "system": sample["system"],
                "input": sample["input"], 
                "output": sample["output"]
            }
            f.write(json.dumps(training_sample, ensure_ascii=False) + '\n')
    
    # 写入验证集
    with open(val_file, 'w', encoding='utf-8') as f:
        for sample in val_samples:
            training_sample = {
                "system": sample["system"],
                "input": sample["input"],
                "output": sample["output"]
            }
            f.write(json.dumps(training_sample, ensure_ascii=False) + '\n')
    
    print(f"训练集: {len(train_samples)} 样本 -> {train_file}")
    print(f"验证集: {len(val_samples)} 样本 -> {val_file}")
    
    # 生成统计信息
    stats = {
        "total_articles": len(articles),
        "total_samples": len(all_samples),
        "train_samples": len(train_samples),
        "val_samples": len(val_samples),
        "train_ratio": train_ratio,
        "years": list(set(meta["year"] for _, meta in articles)),
        "sample_templates": ["改写", "续写", "总结展开"]
    }
    
    stats_file = output_dir / 'dataset_stats.json'
    with open(stats_file, 'w', encoding='utf-8') as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)
    
    print(f"数据集统计: {stats_file}")

def main():
    parser = argparse.ArgumentParser(description="构建风格微调数据集")
    parser.add_argument("--backup_dir", type=str, default="Wechat-Backup", 
                       help="微信备份目录路径")
    parser.add_argument("--output_dir", type=str, default="data",
                       help="输出目录路径")
    parser.add_argument("--train_ratio", type=float, default=0.9,
                       help="训练集比例")
    parser.add_argument("--seed", type=int, default=42,
                       help="随机种子")
    
    args = parser.parse_args()
    
    # 设置随机种子
    random.seed(args.seed)
    
    backup_dir = Path(args.backup_dir)
    output_dir = Path(args.output_dir)
    
    if not backup_dir.exists():
        print(f"备份目录不存在: {backup_dir}")
        return
    
    build_sft_dataset(backup_dir, output_dir, args.train_ratio)
    print("数据集构建完成！")

if __name__ == "__main__":
    main()