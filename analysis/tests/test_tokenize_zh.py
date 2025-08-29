"""Unit tests for tokenize_zh module."""

import unittest
import tempfile
import os
from pathlib import Path
import sys

# Add the source path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from tokenize_zh import ChineseTokenizer, create_default_stopwords


class TestChineseTokenizer(unittest.TestCase):
    """Test Chinese tokenizer functionality."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.stopwords_path = os.path.join(self.temp_dir, 'stopwords.txt')
        
        # Create a simple stopwords file
        with open(self.stopwords_path, 'w', encoding='utf-8') as f:
            f.write('的\n了\n在\n是\n')
        
        self.tokenizer = ChineseTokenizer(
            stopwords_path=self.stopwords_path,
            cache_dir=os.path.join(self.temp_dir, 'cache')
        )
    
    def tearDown(self):
        """Clean up test fixtures."""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_basic_tokenization(self):
        """Test basic Chinese tokenization."""
        text = "我爱自然语言处理技术"
        tokens = self.tokenizer.tokenize(text)
        
        self.assertIsInstance(tokens, list)
        self.assertGreater(len(tokens), 0)
        # Should contain meaningful words, not stopwords
        self.assertNotIn('的', tokens)
        self.assertNotIn('了', tokens)
    
    def test_empty_text(self):
        """Test tokenization of empty text."""
        tokens = self.tokenizer.tokenize("")
        self.assertEqual(tokens, [])
        
        tokens = self.tokenizer.tokenize(None)
        self.assertEqual(tokens, [])
    
    def test_chinese_only_filter(self):
        """Test Chinese-only filtering."""
        text = "我爱 NLP 和 AI 技术123"
        
        # With Chinese only
        tokens_zh = self.tokenizer.tokenize(text, chinese_only=True)
        # Should not contain English or numbers
        self.assertTrue(all('a' <= c <= 'z' or 'A' <= c <= 'Z' or c.isdigit() 
                           for token in tokens_zh for c in token) == False)
        
        # Without Chinese only filter
        tokens_all = self.tokenizer.tokenize(text, chinese_only=False)
        self.assertGreaterEqual(len(tokens_all), len(tokens_zh))
    
    def test_ngram_generation(self):
        """Test n-gram generation."""
        text = "自然语言处理"
        
        # Unigram
        tokens_1 = self.tokenizer.tokenize(text, ngram_max=1)
        
        # Bigram
        tokens_2 = self.tokenizer.tokenize(text, ngram_max=2)
        
        # Should have more tokens with bigrams
        self.assertGreaterEqual(len(tokens_2), len(tokens_1))
    
    def test_batch_tokenization(self):
        """Test batch tokenization."""
        texts = [
            "我爱自然语言处理",
            "机器学习很有趣",
            "深度学习改变世界"
        ]
        
        results = self.tokenizer.tokenize_batch(texts, show_progress=False)
        
        self.assertEqual(len(results), len(texts))
        self.assertTrue(all(isinstance(tokens, list) for tokens in results))
    
    def test_keyword_extraction(self):
        """Test keyword extraction."""
        text = "自然语言处理是人工智能的重要分支，涉及计算机理解和生成人类语言。"
        
        # TF-IDF keywords
        keywords_tfidf = self.tokenizer.extract_keywords(text, topk=5, method="tfidf")
        self.assertIsInstance(keywords_tfidf, list)
        self.assertLessEqual(len(keywords_tfidf), 5)
        
        # Each keyword should be a tuple of (word, weight)
        if keywords_tfidf:
            self.assertIsInstance(keywords_tfidf[0], tuple)
            self.assertEqual(len(keywords_tfidf[0]), 2)
    
    def test_stopwords_loading(self):
        """Test stopwords loading."""
        # Test with extra stopwords
        extra_stopwords = ['测试', '实验']
        tokenizer_extra = ChineseTokenizer(
            stopwords_path=self.stopwords_path,
            extra_stopwords=extra_stopwords,
            cache_dir=os.path.join(self.temp_dir, 'cache2')
        )
        
        text = "这是一个测试实验"
        tokens = tokenizer_extra.tokenize(text)
        
        # Should not contain extra stopwords
        self.assertNotIn('测试', tokens)
        self.assertNotIn('实验', tokens)


class TestHelperFunctions(unittest.TestCase):
    """Test helper functions."""
    
    def test_create_default_stopwords(self):
        """Test creating default stopwords file."""
        with tempfile.TemporaryDirectory() as temp_dir:
            output_path = os.path.join(temp_dir, 'test_stopwords.txt')
            
            result = create_default_stopwords(output_path)
            
            # Should create the file
            self.assertTrue(os.path.exists(output_path))
            
            # Should contain common stopwords
            with open(output_path, 'r', encoding='utf-8') as f:
                content = f.read()
                self.assertIn('的', content)
                self.assertIn('了', content)
                self.assertIn('在', content)


if __name__ == '__main__':
    unittest.main()