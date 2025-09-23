#!/usr/bin/env python3
"""
Style SFT (Supervised Fine-Tuning) 数据准备工具

功能：
1. 遍历 wechat-backup/**/index.md 及对应的 meta.json
2. 从 index.md 里提取正文段落，过滤掉标题、短句和图片占位符
3. 将正文按 300–800 token 切块
4. 输出两份 JSONL 文件：sft_train.jsonl（90%）、sft_val.jsonl（10%）
5. 每条样本格式为 {"system": "...", "input": "...", "output": "..."}
6. 支持从 meta.json 抽取 year 和 topic 字段，附加在样本里

支持的训练模板：
A) 改写保持语气：用我的口吻改写下面这段材料
B) 续写保持语气：请用我的风格续写下面的文字
C) 总结展开：用我的写作风格，围绕以下内容写一段文字
"""

import os
import re
import json
import yaml
import hashlib
import argparse
from pathlib import Path
from typing import List, Dict, Tuple, Set, Optional
from collections import defaultdict
import random


def load_config(config_path: str) -> Dict:
    """加载配置文件"""
    config_file = Path(config_path)
    if not config_file.exists():
        raise FileNotFoundError(f"配置文件不存在: {config_path}")
    
    try:
        with open(config_file, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        return config
    except yaml.YAMLError as e:
        raise ValueError(f"YAML配置文件格式错误: {e}") from None
    except Exception as e:
        raise Exception(f"配置文件读取失败: {e}") from None


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
    # 去除HTML标签
    content = re.sub(r'<[^>]+>', '', content)
    return content.strip()


def split_into_chunks(text: str, min_length: int = 300, max_length: int = 800) -> List[str]:
    """将文本切分为适当大小的块"""
    chunks = []
    paragraphs = text.split('\n\n')
    current_chunk = ""
    
    for para in paragraphs:
        para = para.strip()
        if not para:
            continue
            
        # 如果当前段落太长，需要进一步切分
        if len(para) > max_length:
            # 按句号、感叹号、问号切分，并保留标点
            sentences = re.findall(r'[^。！？]*[。！？]', para)
            for sentence in sentences:
                sentence = sentence.strip()
                if not sentence:
                    continue
                # 不需要恢复标点，已保留
                
                if len(current_chunk + sentence) > max_length:
                    if len(current_chunk) >= min_length:
                        chunks.append(current_chunk.strip())
                    current_chunk = sentence
                else:
                    current_chunk += sentence
        else:
            if len(current_chunk + para) > max_length:
                if len(current_chunk) >= min_length:
                    chunks.append(current_chunk.strip())
                current_chunk = para
            else:
                if current_chunk:
                    current_chunk += '\n\n' + para
                else:
                    current_chunk = para
    
    # 添加最后一个块
    if current_chunk and len(current_chunk) >= min_length:
        chunks.append(current_chunk.strip())
    
    return [chunk for chunk in chunks if len(chunk) >= min_length]


def generate_training_templates(chunk: str, meta_info: Dict, config: Dict) -> List[Dict]:
    """为文本块生成多种训练模板"""
    templates = []
    system_prompt = "你是该公众号作者，保持其叙述节奏与转折。"
    
    # 模板A：改写指令
    if config.get('enable_rewrite_template', True):
        templates.append({
            "system": system_prompt,
            "input": f"用我的口吻改写下面这段材料：{chunk}",
            "output": chunk,
            "meta": meta_info
        })
    
    # 模板B：续写指令（取前半段，要求续写）
    if config.get('enable_continue_template', True) and len(chunk) > 200:
        split_point = len(chunk) // 2
        # 找到合适的切分点（句号后）
        for i in range(split_point, min(split_point + 100, len(chunk))):
            if chunk[i] in '。！？':
                split_point = i + 1
                break
        
        prefix = chunk[:split_point].strip()
        
        if prefix:
            templates.append({
                "system": system_prompt,
                "input": f"请用我的风格续写下面的文字：{prefix}",
                "output": chunk,
                "meta": meta_info
            })
    
    # 模板C：总结再展开
    if config.get('enable_summarize_template', True) and len(chunk) > 400:
        templates.append({
            "system": system_prompt,
            "input": f"用我的写作风格，围绕以下内容写一段文字：{chunk[:100]}...",
            "output": chunk,
            "meta": meta_info
        })
    
    return templates


def calculate_text_hash(text: str) -> str:
    """计算文本的简单哈希，用于去重"""
    return hashlib.md5(text.encode('utf-8')).hexdigest()


def extract_topic_from_path(file_path: str) -> str:
    """从文件路径提取主题信息"""
    # 简单的主题提取逻辑，可以根据需要扩展
    path_parts = Path(file_path).parts
    if len(path_parts) > 2:
        return path_parts[-2]  # 使用倒数第二个路径部分作为主题
    return "其他"


def load_wechat_backup_data(backup_dir: Path, config: Dict) -> List[Tuple[str, Dict]]:
    """加载微信备份数据"""
    articles = []
    filter_years = config.get('filter_years')
    min_article_length = config.get('min_article_length', 100)
    max_articles = config.get('max_articles')
    verbose = config.get('verbose', True)
    
    processed_count = 0
    
    if not backup_dir.exists():
        raise FileNotFoundError(f"备份目录不存在: {backup_dir}")
    
    # 查找所有 markdown 文件
    md_files = list(backup_dir.rglob("*.md"))
    
    # 过滤掉不需要的文件
    md_files = [f for f in md_files if f.name not in ['目录.md', '合集.md', 'action_suggestion.md', 'README.md']]
    
    if verbose:
        print(f"找到 {len(md_files)} 个候选文件")
    
    for md_file in md_files:
        if max_articles and processed_count >= max_articles:
            break
            
        try:
            # 读取 Markdown 内容
            with open(md_file, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # 查找对应的 meta.json
            meta_file = md_file.parent / 'meta.json'
            meta_info = {}
            
            if meta_file.exists():
                try:
                    with open(meta_file, 'r', encoding='utf-8') as f:
                        meta_info = json.load(f)
                except json.JSONDecodeError as e:
                    if verbose:
                        print(f"警告: meta.json 格式错误 {meta_file}: {e}")
                    continue
            
            # 年份过滤
            year = meta_info.get('year', 'unknown')
            if filter_years and year not in filter_years:
                continue
            
            # 添加路径和主题信息
            meta_info['file_path'] = str(md_file.relative_to(backup_dir))
            meta_info['year'] = year
            meta_info['topic'] = meta_info.get('topic', extract_topic_from_path(meta_info['file_path']))
            
            # 清理内容
            cleaned_content = clean_markdown_content(content)
            
            # 过滤太短的内容
            if len(cleaned_content) >= min_article_length:
                articles.append((cleaned_content, meta_info))
                processed_count += 1
                
                if verbose and processed_count % 50 == 0:
                    print(f"已处理 {processed_count} 篇文章...")
                    
        except Exception as e:
            if verbose:
                print(f"处理文件 {md_file} 时出错: {e}")
            continue
    
    if verbose:
        print(f"最终处理了 {len(articles)} 篇文章")
    
    return articles


def build_sft_dataset(config: Dict):
    """构建 SFT 数据集"""
    # 解析路径配置，支持相对和绝对路径
    script_dir = Path(__file__).parent
    
    # 输入目录路径解析
    input_dir_str = config['input_dir']
    if not Path(input_dir_str).is_absolute():
        input_dir = script_dir / input_dir_str
    else:
        input_dir = Path(input_dir_str)
    input_dir = input_dir.resolve()
    
    # 输出路径解析
    train_output = config['train_output']
    val_output = config['val_output']
    stats_output = config['stats_output']
    
    # 如果输出路径是相对路径，相对于脚本目录
    if not Path(train_output).is_absolute():
        train_output = script_dir / train_output
    if not Path(val_output).is_absolute():
        val_output = script_dir / val_output  
    if not Path(stats_output).is_absolute():
        stats_output = script_dir / stats_output
    
    val_split = config.get('val_split', 0.1)
    random_seed = config.get('random_seed', 42)
    enable_dedup = config.get('enable_deduplication', True)
    verbose = config.get('verbose', True)
    
    # 设置随机种子
    random.seed(random_seed)
    
    if verbose:
        print(f"扫描备份目录: {input_dir}")
    
    # 加载数据
    articles = load_wechat_backup_data(input_dir, config)
    
    if not articles:
        print("未找到任何文章，请检查配置")
        return
    
    # 生成训练样本
    all_samples = []
    seen_hashes = set() if enable_dedup else None
    
    chunk_config = {
        'min_length': config.get('min_chunk_length', 300),
        'max_length': config.get('max_chunk_length', 800)
    }
    
    for content, meta_info in articles:
        chunks = split_into_chunks(content, chunk_config['min_length'], chunk_config['max_length'])
        
        if verbose:
            title = meta_info.get('title', '无标题')
            print(f"文章 {title} 切分为 {len(chunks)} 个块")
        
        for chunk in chunks:
            # 去重检查
            if enable_dedup:
                chunk_hash = calculate_text_hash(chunk)
                if chunk_hash in seen_hashes:
                    continue
                seen_hashes.add(chunk_hash)
            
            # 生成训练模板
            templates = generate_training_templates(chunk, meta_info, config)
            all_samples.extend(templates)
    
    if verbose:
        print(f"生成 {len(all_samples)} 个训练样本")
    
    # 打乱并分割数据集
    random.shuffle(all_samples)
    split_point = int(len(all_samples) * (1 - val_split))
    
    train_samples = all_samples[:split_point]
    val_samples = all_samples[split_point:]
    
    # 确保输出目录存在
    for output_path in [train_output, val_output, stats_output]:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    
    # 写入训练集
    try:
        with open(train_output, 'w', encoding='utf-8') as f:
            for sample in train_samples:
                # 移除 meta 信息，只保留训练需要的字段
                training_sample = {
                    "system": sample["system"],
                    "input": sample["input"], 
                    "output": sample["output"]
                }
                f.write(json.dumps(training_sample, ensure_ascii=False) + '\n')
    except Exception as e:
        raise Exception(f"写入训练集文件失败: {e}") from None
    
    # 写入验证集
    try:
        with open(val_output, 'w', encoding='utf-8') as f:
            for sample in val_samples:
                training_sample = {
                    "system": sample["system"],
                    "input": sample["input"],
                    "output": sample["output"]
                }
                f.write(json.dumps(training_sample, ensure_ascii=False) + '\n')
    except Exception as e:
        raise Exception(f"写入验证集文件失败: {e}") from None
    
    # 生成统计信息
    years = list(set(meta.get("year") for _, meta in articles if meta.get("year") is not None))
    topics = list(set(meta.get("topic") for _, meta in articles if meta.get("topic") is not None))
    
    stats = {
        "total_articles": len(articles),
        "total_samples": len(all_samples),
        "train_samples": len(train_samples),
        "val_samples": len(val_samples),
        "val_split": val_split,
        "years": sorted(years),
        "topics": sorted(topics),
        "sample_templates": [],
        "config_used": {k: v for k, v in config.items() if k not in ['input_dir']}  # 排除路径信息
    }
    
    # 统计启用的模板类型
    if config.get('enable_rewrite_template', True):
        stats["sample_templates"].append("改写")
    if config.get('enable_continue_template', True):
        stats["sample_templates"].append("续写")
    if config.get('enable_summarize_template', True):
        stats["sample_templates"].append("总结展开")
    
    try:
        with open(stats_output, 'w', encoding='utf-8') as f:
            json.dump(stats, f, ensure_ascii=False, indent=2)
    except Exception as e:
        raise Exception(f"写入统计文件失败: {e}") from None
    
    if verbose:
        print(f"训练集: {len(train_samples)} 样本 -> {train_output}")
        print(f"验证集: {len(val_samples)} 样本 -> {val_output}")
        print(f"数据集统计: {stats_output}")
        print("数据集构建完成！")


def main():
    parser = argparse.ArgumentParser(description="Style SFT 数据准备工具")
    parser.add_argument("--config", type=str, default="sample_config.yaml",
                       help="配置文件路径")
    parser.add_argument("--input_dir", type=str,
                       help="输入目录路径（覆盖配置文件）")
    parser.add_argument("--output_dir", type=str,
                       help="输出目录路径（覆盖配置文件）")
    
    args = parser.parse_args()
    
    try:
        # 加载配置
        config = load_config(args.config)
        
        # 命令行参数覆盖
        if args.input_dir:
            config['input_dir'] = args.input_dir
        
        if args.output_dir:
            config['train_output'] = f"{args.output_dir}/sft_train.jsonl"
            config['val_output'] = f"{args.output_dir}/sft_val.jsonl"
            config['stats_output'] = f"{args.output_dir}/dataset_stats.json"
        
        # 验证必要的配置
        required_keys = ['input_dir', 'train_output', 'val_output', 'stats_output']
        for key in required_keys:
            if key not in config:
                print(f"配置文件缺少必要字段: {key}")
                return 1
        
        # 构建数据集
        build_sft_dataset(config)
        return 0
        
    except FileNotFoundError as e:
        print(f"错误: {e}")
        return 1
    except ValueError as e:
        print(f"配置错误: {e}")
        return 1
    except Exception as e:
        print(f"处理过程中出错: {e}")
        return 1


if __name__ == "__main__":
    main()