"""
Simplified data loader that works with standard library only.
This is a minimal version for demonstration when external packages are not available.
"""

import json
import logging
import os
from pathlib import Path
from typing import Dict, List, Optional, Union, Any
import csv
import re

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def detect_encoding_simple(file_path: Path) -> str:
    """
    Simple encoding detection using common encodings.
    """
    common_encodings = ['utf-8', 'utf-8-sig', 'gbk', 'gb2312']
    
    for encoding in common_encodings:
        try:
            with open(file_path, 'r', encoding=encoding) as f:
                f.read(1024)  # Try reading first 1KB
            return encoding
        except (UnicodeDecodeError, UnicodeError):
            continue
    
    # Default fallback
    return 'utf-8'


def load_json_file_simple(file_path: Path) -> List[Dict[str, Any]]:
    """
    Load JSON file with simple encoding detection.
    """
    encoding = detect_encoding_simple(file_path)
    logger.info(f"检测到文件 {file_path.name} 编码: {encoding}")
    
    try:
        with open(file_path, 'r', encoding=encoding) as f:
            data = json.load(f)
        
        # Handle different JSON structures
        if isinstance(data, list):
            return data
        elif isinstance(data, dict):
            # Check for common nested structures
            for key in ['data', 'articles', 'items', 'records']:
                if key in data and isinstance(data[key], list):
                    return data[key]
            # If single object, wrap in list
            return [data]
        else:
            raise ValueError(f"不支持的JSON结构类型: {type(data)}")
            
    except json.JSONDecodeError as e:
        raise Exception(f"JSON解析失败: {e}")
    except Exception as e:
        raise Exception(f"文件读取失败: {e}")


def load_csv_file_simple(file_path: Path) -> List[Dict[str, Any]]:
    """
    Load CSV file with simple encoding detection.
    """
    encoding = detect_encoding_simple(file_path)
    logger.info(f"检测到文件 {file_path.name} 编码: {encoding}")
    
    try:
        records = []
        with open(file_path, 'r', encoding=encoding, newline='') as f:
            # Try to detect delimiter
            sample = f.read(1024)
            f.seek(0)
            
            delimiter = ','
            if '\t' in sample:
                delimiter = '\t'
            elif ';' in sample:
                delimiter = ';'
            
            reader = csv.DictReader(f, delimiter=delimiter)
            for row in reader:
                records.append(dict(row))
        
        return records
        
    except Exception as e:
        raise Exception(f"CSV文件读取失败: {e}")


def normalize_column_names_simple(record: Dict[str, Any]) -> Dict[str, Any]:
    """
    Normalize column names to standard schema.
    """
    column_mapping = {
        # English variants
        'title': 'title',
        'content': 'content', 
        'text': 'content',
        'body': 'content',
        'article': 'content',
        'url': 'url',
        'link': 'url',
        'href': 'url',
        'date': 'date',
        'published_at': 'date',
        'publish_date': 'date',
        'created_at': 'date',
        'time': 'date',
        
        # Chinese variants
        '标题': 'title',
        '题目': 'title',
        '内容': 'content',
        '正文': 'content',
        '文本': 'content',
        '文章': 'content',
        '链接': 'url',
        '网址': 'url',
        'URL': 'url',
        '日期': 'date',
        '时间': 'date',
        '发布时间': 'date',
        '创建时间': 'date',
    }
    
    # Create mapping for actual columns (case-insensitive)
    normalized = {}
    for key, value in record.items():
        key_lower = key.lower().strip()
        mapped_key = None
        
        for pattern, standard in column_mapping.items():
            if key_lower == pattern.lower():
                mapped_key = standard
                break
        
        if mapped_key:
            normalized[mapped_key] = value
        else:
            normalized[key] = value
    
    # If there's no 'content' field, try to find a text field
    if 'content' not in normalized:
        for key, value in normalized.items():
            if isinstance(value, str) and len(value) > 10:
                normalized['content'] = value
                break
    
    return normalized


def standardize_record_simple(record: Dict[str, Any], source_file: str) -> Dict[str, Optional[str]]:
    """
    Standardize a record to the unified schema.
    """
    def safe_str(value) -> Optional[str]:
        """Convert value to string, handling None"""
        if value is None or value == '':
            return None
        return str(value).strip() if str(value).strip() else None
    
    # Normalize column names first
    normalized = normalize_column_names_simple(record)
    
    standardized = {
        'title': safe_str(normalized.get('title')),
        'content': safe_str(normalized.get('content')),
        'url': safe_str(normalized.get('url')),
        'date': safe_str(normalized.get('date')),
        'source_file': source_file
    }
    
    return standardized


def load_corpus_simple(data_dir: Union[str, Path]) -> List[Dict[str, Optional[str]]]:
    """
    Load corpus from directory, supporting JSON and CSV files.
    Simplified version using only standard library.
    """
    data_path = Path(data_dir).resolve()
    
    if not data_path.exists():
        raise FileNotFoundError(f"数据目录不存在: {data_path}")
    
    if not data_path.is_dir():
        raise FileNotFoundError(f"路径不是目录: {data_path}")
    
    # Find all supported files recursively
    supported_extensions = {'.json', '.csv'}
    files = [f for f in data_path.rglob('*') 
             if f.is_file() and f.suffix.lower() in supported_extensions]
    
    if not files:
        raise Exception(f"在目录 {data_path} 中未找到支持的文件 (*.json, *.csv)")
    
    records = []
    failed_files = []
    
    for file_path in files:
        logger.info(f"正在处理文件: {file_path.name}")
        
        try:
            # Check if file is empty
            if file_path.stat().st_size == 0:
                logger.warning(f"跳过空文件: {file_path.name}")
                failed_files.append((str(file_path), "文件为空"))
                continue
            
            # Load based on extension
            if file_path.suffix.lower() == '.json':
                file_records = load_json_file_simple(file_path)
            elif file_path.suffix.lower() == '.csv':
                file_records = load_csv_file_simple(file_path)
            else:
                continue
            
            # Standardize records
            for record in file_records:
                standardized = standardize_record_simple(record, file_path.name)
                if standardized['content']:  # Only add records with content
                    records.append(standardized)
            
            logger.info(f"成功加载 {len(file_records)} 条记录从 {file_path.name}")
            
        except Exception as e:
            error_msg = str(e)
            logger.error(f"文件 {file_path.name} 加载失败: {error_msg}")
            failed_files.append((str(file_path), error_msg))
    
    if not records:
        raise Exception("未能从任何文件中加载到有效记录")
    
    logger.info(f"总共加载 {len(records)} 条有效记录")
    
    # Store failed files info for UI display
    if hasattr(load_corpus_simple, '_failed_files'):
        load_corpus_simple._failed_files = failed_files
    else:
        setattr(load_corpus_simple, '_failed_files', failed_files)
    
    return records


def summarize_load_stats_simple(records: List[Dict[str, Optional[str]]]) -> Dict[str, Any]:
    """
    Generate loading statistics summary.
    """
    if not records:
        return {
            'total_files': 0,
            'total_records': 0,
            'empty_content': 0,
            'avg_content_length': 0,
            'failed_files': getattr(load_corpus_simple, '_failed_files', [])
        }
    
    # Count source files
    source_files = set(record['source_file'] for record in records)
    
    # Count empty content
    empty_content = sum(1 for record in records if not record['content'])
    
    # Calculate average content length
    content_lengths = [len(record['content']) for record in records if record['content']]
    avg_length = sum(content_lengths) / len(content_lengths) if content_lengths else 0
    
    return {
        'total_files': len(source_files),
        'total_records': len(records),
        'empty_content': empty_content,
        'avg_content_length': round(avg_length, 1),
        'failed_files': getattr(load_corpus_simple, '_failed_files', [])
    }


# Simple text processing functions

def clean_text_simple(text: str) -> str:
    """
    Clean text by removing URLs, emails, excessive whitespace.
    """
    if not text:
        return ""
    
    # Remove URLs
    text = re.sub(r'https?://[^\s]+', '', text)
    text = re.sub(r'www\.[^\s]+', '', text)
    
    # Remove email addresses
    text = re.sub(r'\S+@\S+', '', text)
    
    # Remove excessive whitespace
    text = re.sub(r'\s+', ' ', text)
    
    # Remove standalone numbers
    text = re.sub(r'\b\d+\b', ' ', text)
    
    return text.strip()


def simple_chinese_tokenize(text: str) -> List[str]:
    """
    Simple Chinese tokenization by character splitting with basic filtering.
    """
    if not text:
        return []
    
    # Clean text first
    text = clean_text_simple(text)
    
    # Split by characters and filter
    tokens = []
    current_word = ""
    
    for char in text:
        if '\u4e00' <= char <= '\u9fff':  # Chinese character
            if current_word:
                tokens.append(current_word)
                current_word = ""
            tokens.append(char)
        elif char.isalpha():  # English letter
            current_word += char
        else:  # Punctuation or space
            if current_word:
                tokens.append(current_word)
                current_word = ""
    
    if current_word:
        tokens.append(current_word)
    
    # Filter out very short tokens
    return [token for token in tokens if len(token.strip()) >= 1]


def compute_frequencies_simple(tokens: List[str]) -> List[Dict[str, Any]]:
    """
    Compute token frequencies and proportions using standard library.
    """
    if not tokens:
        return []
    
    # Count frequencies
    freq_dict = {}
    for token in tokens:
        freq_dict[token] = freq_dict.get(token, 0) + 1
    
    # Convert to list and sort
    total_tokens = len(tokens)
    freq_data = []
    
    for term, freq in sorted(freq_dict.items(), key=lambda x: x[1], reverse=True):
        prop = freq / total_tokens
        freq_data.append({
            'term': term,
            'freq': freq,
            'prop': round(prop, 6)
        })
    
    return freq_data


if __name__ == "__main__":
    # Test with default data directory
    try:
        data_dir = Path(__file__).parent / "data"
        records = load_corpus_simple(data_dir)
        stats = summarize_load_stats_simple(records)
        print(f"加载统计: {stats}")
        if records:
            print(f"示例记录: {records[0]}")
            
            # Test tokenization
            all_text = " ".join([r['content'] for r in records if r['content']])
            tokens = simple_chinese_tokenize(all_text)
            freq_data = compute_frequencies_simple(tokens)
            print(f"词频统计前10: {freq_data[:10]}")
            
    except Exception as e:
        print(f"测试失败: {e}")