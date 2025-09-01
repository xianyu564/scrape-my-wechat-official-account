"""
Unit tests for data_loader module.
"""

import json
import tempfile
import unittest
from pathlib import Path
import pandas as pd

from data_loader import (
    load_corpus, detect_encoding, normalize_column_names,
    load_json_file, load_csv_file, load_parquet_file,
    standardize_record, summarize_load_stats
)


class TestDataLoader(unittest.TestCase):
    
    def setUp(self):
        """Set up test fixtures"""
        self.temp_dir = Path(tempfile.mkdtemp())
    
    def tearDown(self):
        """Clean up test fixtures"""
        import shutil
        shutil.rmtree(self.temp_dir)
    
    def test_load_empty_directory(self):
        """Test loading from empty directory"""
        empty_dir = self.temp_dir / "empty"
        empty_dir.mkdir()
        
        with self.assertRaises(Exception) as context:
            load_corpus(empty_dir)
        
        self.assertIn("未找到支持的文件", str(context.exception))
    
    def test_load_nonexistent_directory(self):
        """Test loading from nonexistent directory"""
        nonexistent_dir = self.temp_dir / "nonexistent"
        
        with self.assertRaises(FileNotFoundError):
            load_corpus(nonexistent_dir)
    
    def test_load_empty_file(self):
        """Test loading empty files"""
        # Create empty JSON file
        empty_json = self.temp_dir / "empty.json"
        empty_json.touch()
        
        # Should handle empty files gracefully
        try:
            records = load_corpus(self.temp_dir)
            # Should not raise exception but may return empty list
        except Exception as e:
            # Should give informative error message
            self.assertIn("empty", str(e).lower())
    
    def test_load_invalid_encoding(self):
        """Test loading files with problematic encoding"""
        # Create file with mixed encoding content
        bad_encoding_file = self.temp_dir / "bad_encoding.json"
        
        # Write some problematic content
        with open(bad_encoding_file, 'wb') as f:
            # Mix of UTF-8 and invalid bytes
            f.write(b'{"title": "\xe4\xb8\xad\xe6\x96\x87", "content": "test\xff\xfe"}')
        
        # Should handle encoding issues gracefully
        try:
            records = load_corpus(self.temp_dir)
        except Exception as e:
            self.assertIn("编码", str(e))
    
    def test_load_missing_columns(self):
        """Test loading files with missing required columns"""
        # Create CSV with missing content column
        missing_cols_csv = self.temp_dir / "missing_cols.csv"
        df = pd.DataFrame({
            'title': ['标题1', '标题2'],
            'url': ['http://example1.com', 'http://example2.com']
            # Missing 'content' column
        })
        df.to_csv(missing_cols_csv, index=False, encoding='utf-8')
        
        # Should handle missing columns gracefully
        records = load_corpus(self.temp_dir)
        # May return empty list or records with None content
        if records:
            self.assertTrue(all(r['content'] is None for r in records))
    
    def test_load_mixed_formats(self):
        """Test loading directory with mixed file formats"""
        # Create JSON file
        json_file = self.temp_dir / "test.json"
        json_data = [
            {"title": "JSON标题1", "content": "JSON内容1", "url": "http://json1.com"},
            {"title": "JSON标题2", "content": "JSON内容2", "url": "http://json2.com"}
        ]
        with open(json_file, 'w', encoding='utf-8') as f:
            json.dump(json_data, f, ensure_ascii=False)
        
        # Create CSV file
        csv_file = self.temp_dir / "test.csv"
        csv_data = pd.DataFrame({
            'title': ['CSV标题1', 'CSV标题2'],
            'content': ['CSV内容1', 'CSV内容2'],
            'url': ['http://csv1.com', 'http://csv2.com']
        })
        csv_data.to_csv(csv_file, index=False, encoding='utf-8')
        
        # Create Parquet file
        parquet_file = self.temp_dir / "test.parquet"
        parquet_data = pd.DataFrame({
            'title': ['Parquet标题1', 'Parquet标题2'],
            'content': ['Parquet内容1', 'Parquet内容2'],
            'url': ['http://parquet1.com', 'http://parquet2.com']
        })
        parquet_data.to_parquet(parquet_file, index=False)
        
        # Load all files
        records = load_corpus(self.temp_dir)
        
        # Should load 6 records total (2 from each file)
        self.assertEqual(len(records), 6)
        
        # Check that all records have required fields
        for record in records:
            self.assertIn('title', record)
            self.assertIn('content', record)
            self.assertIn('url', record)
            self.assertIn('source_file', record)
    
    def test_column_normalization(self):
        """Test column name normalization"""
        # Test various column name variants
        test_cases = [
            # English variants
            {'Title': 'test', 'Content': 'content', 'URL': 'url', 'Date': 'date'},
            {'title': 'test', 'text': 'content', 'link': 'url', 'published_at': 'date'},
            
            # Chinese variants
            {'标题': 'test', '内容': 'content', '链接': 'url', '日期': 'date'},
            {'题目': 'test', '正文': 'content', '网址': 'url', '时间': 'date'},
        ]
        
        for test_data in test_cases:
            df = pd.DataFrame([test_data])
            normalized = normalize_column_names(df)
            
            # Should have standard column names
            self.assertIn('title', normalized.columns)
            self.assertIn('content', normalized.columns)
            self.assertIn('url', normalized.columns)
            self.assertIn('date', normalized.columns)
    
    def test_single_column_text(self):
        """Test files with only single text column"""
        # Create CSV with only one text column
        single_col_csv = self.temp_dir / "single_col.csv"
        df = pd.DataFrame({
            'text_content': [
                '这是第一段长文本内容，应该被识别为content字段。',
                '这是第二段长文本内容，也应该被识别为content字段。'
            ]
        })
        df.to_csv(single_col_csv, index=False, encoding='utf-8')
        
        records = load_corpus(self.temp_dir)
        
        # Should map the text column to content
        self.assertTrue(len(records) >= 2)
        for record in records:
            if record['source_file'] == 'single_col.csv':
                self.assertIsNotNone(record['content'])
                self.assertIn('长文本内容', record['content'])
    
    def test_summarize_stats(self):
        """Test statistics summarization"""
        # Create test data
        records = [
            {'title': '标题1', 'content': '内容1', 'url': 'url1', 'date': None, 'source_file': 'file1.json'},
            {'title': '标题2', 'content': '内容2很长很长很长', 'url': None, 'date': 'date2', 'source_file': 'file1.json'},
            {'title': None, 'content': None, 'url': 'url3', 'date': 'date3', 'source_file': 'file2.csv'},
            {'title': '标题4', 'content': '内容4', 'url': 'url4', 'date': 'date4', 'source_file': 'file2.csv'},
        ]
        
        stats = summarize_load_stats(records)
        
        self.assertEqual(stats['total_files'], 2)  # file1.json, file2.csv
        self.assertEqual(stats['total_records'], 4)
        self.assertEqual(stats['empty_content'], 1)  # One record with None content
        self.assertGreater(stats['avg_content_length'], 0)
    
    def test_standardize_record(self):
        """Test record standardization"""
        # Test with complete record
        complete_record = {
            'title': '测试标题',
            'content': '测试内容',
            'url': 'http://test.com',
            'date': '2024-01-01'
        }
        
        standardized = standardize_record(complete_record, 'test.json')
        
        self.assertEqual(standardized['title'], '测试标题')
        self.assertEqual(standardized['content'], '测试内容')
        self.assertEqual(standardized['url'], 'http://test.com')
        self.assertEqual(standardized['date'], '2024-01-01')
        self.assertEqual(standardized['source_file'], 'test.json')
        
        # Test with missing fields
        incomplete_record = {
            'some_text': '这是一些文本内容应该被映射到content'
        }
        
        standardized = standardize_record(incomplete_record, 'test.json')
        
        self.assertIsNone(standardized['title'])
        self.assertEqual(standardized['content'], '这是一些文本内容应该被映射到content')
        self.assertIsNone(standardized['url'])
        self.assertIsNone(standardized['date'])


if __name__ == '__main__':
    unittest.main()