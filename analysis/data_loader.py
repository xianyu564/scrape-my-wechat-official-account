"""
Data loader module for WeChat article corpus.
Handles multiple file formats (JSON, CSV, Parquet) with robust encoding detection.
"""

import json
import logging
import os
from pathlib import Path
from typing import Dict, List, Optional, Union, Any

import chardet
import pandas as pd
from charset_normalizer import from_path

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def detect_encoding(file_path: Path) -> str:
    """
    Detect file encoding using multiple methods.
    
    Args:
        file_path: Path to the file
        
    Returns:
        Detected encoding string
        
    Raises:
        UnicodeDecodeError: If all encoding detection methods fail
    """
    try:
        # Try charset-normalizer first (more accurate)
        result = from_path(file_path).best()
        if result:
            return str(result.encoding)
    except Exception:
        pass
    
    try:
        # Fallback to chardet
        with open(file_path, 'rb') as f:
            raw_data = f.read()
            result = chardet.detect(raw_data)
            if result['encoding']:
                return result['encoding']
    except Exception:
        pass
    
    # Try common encodings as last resort
    common_encodings = ['utf-8', 'utf-8-sig', 'gbk', 'gb2312', 'big5']
    for encoding in common_encodings:
        try:
            with open(file_path, 'r', encoding=encoding) as f:
                f.read(1024)  # Try reading first 1KB
            return encoding
        except (UnicodeDecodeError, UnicodeError):
            continue
    
    raise UnicodeDecodeError(f"无法检测文件 {file_path} 的编码格式")


def normalize_column_names(df: pd.DataFrame) -> pd.DataFrame:
    """
    Normalize column names to standard schema.
    Supports both Chinese and English column names.
    
    Args:
        df: Input DataFrame
        
    Returns:
        DataFrame with normalized column names
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
    actual_mapping = {}
    for col in df.columns:
        col_lower = col.lower().strip()
        for key, value in column_mapping.items():
            if col_lower == key.lower():
                actual_mapping[col] = value
                break
    
    # Rename columns
    df_renamed = df.rename(columns=actual_mapping)
    
    # If there's only one text column and no 'content', map it to content
    if 'content' not in df_renamed.columns:
        text_columns = [col for col in df_renamed.columns 
                       if df_renamed[col].dtype == 'object' and 
                       df_renamed[col].str.len().mean() > 10]  # Assume content columns are longer
        if text_columns:
            df_renamed = df_renamed.rename(columns={text_columns[0]: 'content'})
    
    return df_renamed


def load_json_file(file_path: Path) -> List[Dict[str, Any]]:
    """
    Load JSON file with encoding detection.
    
    Args:
        file_path: Path to JSON file
        
    Returns:
        List of records
        
    Raises:
        Exception: If file cannot be loaded or parsed
    """
    encoding = detect_encoding(file_path)
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


def load_csv_file(file_path: Path) -> List[Dict[str, Any]]:
    """
    Load CSV file with encoding detection.
    
    Args:
        file_path: Path to CSV file
        
    Returns:
        List of records
        
    Raises:
        Exception: If file cannot be loaded or parsed
    """
    encoding = detect_encoding(file_path)
    logger.info(f"检测到文件 {file_path.name} 编码: {encoding}")
    
    try:
        # Try different separators
        separators = [',', '\t', ';', '|']
        df = None
        
        for sep in separators:
            try:
                df = pd.read_csv(file_path, encoding=encoding, separator=sep)
                if len(df.columns) > 1:  # Multi-column file found
                    break
            except Exception:
                continue
        
        if df is None or df.empty:
            raise Exception("无法解析CSV文件或文件为空")
        
        # Normalize column names
        df = normalize_column_names(df)
        
        # Convert to records
        return df.to_dict('records')
        
    except Exception as e:
        raise Exception(f"CSV文件读取失败: {e}")


def load_parquet_file(file_path: Path) -> List[Dict[str, Any]]:
    """
    Load Parquet file.
    
    Args:
        file_path: Path to Parquet file
        
    Returns:
        List of records
        
    Raises:
        Exception: If file cannot be loaded
    """
    try:
        df = pd.read_parquet(file_path)
        
        if df.empty:
            raise Exception("Parquet文件为空")
        
        # Normalize column names
        df = normalize_column_names(df)
        
        # Convert to records
        return df.to_dict('records')
        
    except Exception as e:
        raise Exception(f"Parquet文件读取失败: {e}")


def standardize_record(record: Dict[str, Any], source_file: str) -> Dict[str, Optional[str]]:
    """
    Standardize a record to the unified schema.
    
    Args:
        record: Input record dictionary
        source_file: Source file name
        
    Returns:
        Standardized record with keys: title, content, url, date, source_file
    """
    def safe_str(value) -> Optional[str]:
        """Convert value to string, handling None and NaN"""
        if pd.isna(value) if hasattr(pd, 'isna') else value is None:
            return None
        return str(value).strip() if str(value).strip() else None
    
    standardized = {
        'title': safe_str(record.get('title')),
        'content': safe_str(record.get('content')),
        'url': safe_str(record.get('url')),
        'date': safe_str(record.get('date')),
        'source_file': source_file
    }
    
    # Ensure content is not None or empty
    if not standardized['content']:
        # Try to find any text content in the record
        for key, value in record.items():
            if isinstance(value, str) and len(value.strip()) > 10:
                standardized['content'] = value.strip()
                break
    
    return standardized


def load_corpus(data_dir: Union[str, Path]) -> List[Dict[str, Optional[str]]]:
    """
    Load corpus from directory, supporting JSON, CSV, and Parquet files.
    
    Args:
        data_dir: Directory containing data files
        
    Returns:
        List of standardized records
        
    Raises:
        FileNotFoundError: If directory doesn't exist
        Exception: If no valid files found
        Exception: If data_dir is outside the allowed root
    """
    # Define safe root directory
    safe_root = (Path(__file__).parent / "data").resolve()
    data_path = Path(data_dir).resolve()
    # Validate that the user-supplied path is strictly contained within safe_root
    try:
        data_path.relative_to(safe_root)
    except ValueError:
        raise Exception(f"数据目录超出允许范围: {data_path}")
    
    if not data_path.exists():
        raise FileNotFoundError(f"数据目录不存在: {data_path}")
    
    if not data_path.is_dir():
        raise FileNotFoundError(f"路径不是目录: {data_path}")
    
    # Find all supported files recursively
    supported_extensions = {'.json', '.csv', '.parquet'}
    files = [f for f in data_path.rglob('*') 
             if f.is_file() and f.suffix.lower() in supported_extensions]
    
    if not files:
        raise Exception(f"在目录 {data_path} 中未找到支持的文件 (*.json, *.csv, *.parquet)")
    
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
                file_records = load_json_file(file_path)
            elif file_path.suffix.lower() == '.csv':
                file_records = load_csv_file(file_path)
            elif file_path.suffix.lower() == '.parquet':
                file_records = load_parquet_file(file_path)
            else:
                continue
            
            # Standardize records
            for record in file_records:
                standardized = standardize_record(record, file_path.name)
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
    if hasattr(load_corpus, '_failed_files'):
        load_corpus._failed_files = failed_files
    else:
        setattr(load_corpus, '_failed_files', failed_files)
    
    return records


def summarize_load_stats(records: List[Dict[str, Optional[str]]]) -> Dict[str, Any]:
    """
    Generate loading statistics summary.
    
    Args:
        records: List of loaded records
        
    Returns:
        Dictionary with statistics
    """
    if not records:
        return {
            'total_files': 0,
            'total_records': 0,
            'empty_content': 0,
            'avg_content_length': 0,
            'failed_files': getattr(load_corpus, '_failed_files', [])
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
        'failed_files': getattr(load_corpus, '_failed_files', [])
    }


if __name__ == "__main__":
    # Test with default data directory
    try:
        data_dir = Path(__file__).parent / "data"
        records = load_corpus(data_dir)
        stats = summarize_load_stats(records)
        print(f"加载统计: {stats}")
        if records:
            print(f"示例记录: {records[0]}")
    except Exception as e:
        print(f"测试失败: {e}")