"""
Unit tests for text_pipeline module.
"""

import unittest
from text_pipeline import (
    tokenize, ngram_tokenize, compute_frequencies, kwic_index,
    filter_tokens, apply_synonym_mapping, clean_text,
    apply_traditional_simplified_conversion
)


class TestTextPipeline(unittest.TestCase):
    
    def setUp(self):
        """Set up test fixtures"""
        self.test_records = [
            {'content': '这是一个测试文本，包含中文和English混合内容。人工智能很有趣。'},
            {'content': '机器学习和深度学习是人工智能的重要分支。'},
            {'content': 'Natural language processing (NLP) 自然语言处理很重要。'},
            {'content': '大数据、云计算、人工智能是现代科技的三大支柱。'}
        ]
    
    def test_jieba_exact_tokenization(self):
        """Test jieba exact mode tokenization"""
        tokens = tokenize(self.test_records, mode="jieba_exact")
        
        # Should contain Chinese words
        self.assertIn('人工智能', tokens)
        self.assertIn('机器学习', tokens)
        self.assertIn('深度学习', tokens)
        
        # Should be non-empty
        self.assertGreater(len(tokens), 0)
    
    def test_jieba_search_tokenization(self):
        """Test jieba search mode tokenization"""
        tokens = tokenize(self.test_records, mode="jieba_search")
        
        # Should contain Chinese words
        self.assertIn('人工智能', tokens)
        self.assertIn('机器学习', tokens)
        
        # Search mode typically produces more tokens than exact mode
        exact_tokens = tokenize(self.test_records, mode="jieba_exact")
        self.assertGreaterEqual(len(tokens), len(exact_tokens))
    
    def test_ngram_tokenization_chinese(self):
        """Test n-gram tokenization for Chinese text"""
        chinese_record = [{'content': '人工智能技术发展'}]
        
        # Test different n values
        for n in [1, 2, 3, 4, 5, 6]:
            tokens = tokenize(chinese_record, mode="ngram", n=n)
            self.assertGreater(len(tokens), 0)
            
            # Check n-gram length
            if tokens:
                self.assertEqual(len(tokens[0]), n)
    
    def test_ngram_tokenization_english(self):
        """Test n-gram tokenization for English text"""
        english_record = [{'content': 'natural language processing machine learning'}]
        
        # Test bigrams
        tokens = tokenize(english_record, mode="ngram", n=2)
        self.assertIn('natural language', tokens)
        self.assertIn('language processing', tokens)
        self.assertIn('processing machine', tokens)
        self.assertIn('machine learning', tokens)
    
    def test_large_ngram(self):
        """Test n-gram with n > 4"""
        test_record = [{'content': '人工智能机器学习深度学习神经网络'}]
        
        # Test n=6 (maximum allowed)
        tokens = tokenize(test_record, mode="ngram", n=6)
        self.assertGreater(len(tokens), 0)
        
        # All tokens should be 6 characters long
        for token in tokens:
            self.assertEqual(len(token), 6)
    
    def test_stopwords_filtering(self):
        """Test stopwords filtering"""
        stopwords = {'的', '是', '和', '很', '与'}
        
        tokens = tokenize(
            self.test_records, 
            mode="jieba_exact",
            stopwords=stopwords
        )
        
        # Stopwords should be removed
        for stopword in stopwords:
            self.assertNotIn(stopword, tokens)
    
    def test_whitelist_priority(self):
        """Test that whitelist takes priority over stopwords"""
        stopwords = {'人工智能', '机器学习'}  # Important terms in stopwords
        whitelist = {'人工智能'}  # But whitelist includes 人工智能
        
        tokens = tokenize(
            self.test_records,
            mode="jieba_exact",
            stopwords=stopwords,
            keep_whitelist=whitelist
        )
        
        # Whitelisted term should be kept despite being in stopwords
        self.assertIn('人工智能', tokens)
        # Non-whitelisted stopword should be removed
        self.assertNotIn('机器学习', tokens)
    
    def test_synonym_mapping(self):
        """Test synonym mapping functionality"""
        synonyms = {
            'AI': '人工智能',
            'machine learning': '机器学习',
            'ML': '机器学习',
            'deep learning': '深度学习'
        }
        
        test_record = [{'content': 'AI and ML are important for deep learning research'}]
        
        tokens = tokenize(
            test_record,
            mode="jieba_exact",
            synonyms=synonyms
        )
        
        # Synonyms should be mapped to canonical forms
        self.assertIn('人工智能', tokens)
        self.assertIn('机器学习', tokens)
        self.assertIn('深度学习', tokens)
    
    def test_traditional_simplified_conversion(self):
        """Test traditional to simplified Chinese conversion"""
        traditional_record = [{'content': '機器學習和人工智慧很重要'}]
        
        # Without conversion
        tokens_trad = tokenize(traditional_record, mode="jieba_exact", unify_trad_simp=False)
        
        # With conversion
        tokens_simp = tokenize(traditional_record, mode="jieba_exact", unify_trad_simp=True)
        
        # Results should be different (assuming OpenCC is available)
        if tokens_trad != tokens_simp:
            # Should contain simplified characters
            text_simplified = ' '.join(tokens_simp)
            self.assertIn('机器学习', text_simplified)
            self.assertIn('人工智能', text_simplified)
    
    def test_frequency_computation(self):
        """Test frequency computation"""
        tokens = ['人工智能', '机器学习', '人工智能', '深度学习', '人工智能']
        
        freq_df = compute_frequencies(tokens)
        
        # Check structure
        self.assertListEqual(list(freq_df.columns), ['term', 'freq', 'prop'])
        
        # Check most frequent term
        self.assertEqual(freq_df.iloc[0]['term'], '人工智能')
        self.assertEqual(freq_df.iloc[0]['freq'], 3)
        self.assertAlmostEqual(freq_df.iloc[0]['prop'], 0.6, places=1)
        
        # Check proportions sum to 1
        self.assertAlmostEqual(freq_df['prop'].sum(), 1.0, places=5)
    
    def test_kwic_index(self):
        """Test KWIC (Key Word In Context) functionality"""
        test_records = [
            {
                'title': '文章1',
                'content': '人工智能是现代科技的重要发展方向，人工智能技术应用广泛。',
                'url': 'http://example1.com'
            },
            {
                'title': '文章2', 
                'content': '机器学习作为人工智能的子领域，具有重要意义。',
                'url': 'http://example2.com'
            }
        ]
        
        kwic_results = kwic_index(test_records, '人工智能', window=10)
        
        # Should find multiple occurrences
        self.assertGreaterEqual(len(kwic_results), 2)
        
        # Check structure
        for kwic in kwic_results:
            self.assertIn('left', kwic)
            self.assertIn('keyword', kwic)
            self.assertIn('right', kwic)
            self.assertIn('title', kwic)
            self.assertIn('url', kwic)
            
            # Keyword should match
            self.assertEqual(kwic['keyword'], '人工智能')
    
    def test_clean_text(self):
        """Test text cleaning functionality"""
        dirty_text = """
        这是一个测试文本 https://example.com 包含URL和邮箱 test@example.com
        还有很多    空格和123数字，以及特殊符号！@#$%^&*()
        """
        
        clean = clean_text(dirty_text)
        
        # URLs should be removed
        self.assertNotIn('https://example.com', clean)
        # Email should be removed  
        self.assertNotIn('test@example.com', clean)
        # Excessive whitespace should be normalized
        self.assertNotIn('    ', clean)
        # Chinese text should be preserved
        self.assertIn('测试文本', clean)
    
    def test_filter_tokens(self):
        """Test token filtering with stopwords and whitelist"""
        tokens = ['这', '是', '人工智能', '的', '重要', '应用', '很', '有趣']
        stopwords = {'这', '是', '的', '很'}
        whitelist = {'很'}  # Keep '很' despite it being in stopwords
        
        filtered = filter_tokens(tokens, stopwords, whitelist)
        
        # Whitelist term should be kept
        self.assertIn('很', filtered)
        # Other stopwords should be removed
        self.assertNotIn('这', filtered)
        self.assertNotIn('是', filtered)
        self.assertNotIn('的', filtered)
        # Non-stopwords should be kept
        self.assertIn('人工智能', filtered)
        self.assertIn('重要', filtered)
        self.assertIn('应用', filtered)
        self.assertIn('有趣', filtered)
    
    def test_apply_synonym_mapping(self):
        """Test synonym mapping application"""
        tokens = ['AI', 'machine learning', 'deep learning', 'neural networks']
        synonyms = {
            'AI': '人工智能',
            'machine learning': '机器学习',
            'deep learning': '深度学习'
        }
        
        mapped = apply_synonym_mapping(tokens, synonyms)
        
        # Mapped terms should be replaced
        self.assertIn('人工智能', mapped)
        self.assertIn('机器学习', mapped)
        self.assertIn('深度学习', mapped)
        # Unmapped terms should remain
        self.assertIn('neural networks', mapped)
        # Original terms should not remain
        self.assertNotIn('AI', mapped)
        self.assertNotIn('machine learning', mapped)
    
    def test_empty_input_handling(self):
        """Test handling of empty inputs"""
        # Empty records
        empty_tokens = tokenize([], mode="jieba_exact")
        self.assertEqual(len(empty_tokens), 0)
        
        # Records with empty content
        empty_content_records = [{'content': ''}, {'content': None}]
        tokens = tokenize(empty_content_records, mode="jieba_exact")
        self.assertEqual(len(tokens), 0)
        
        # Empty frequency computation
        empty_freq_df = compute_frequencies([])
        self.assertTrue(empty_freq_df.empty)
        self.assertListEqual(list(empty_freq_df.columns), ['term', 'freq', 'prop'])


if __name__ == '__main__':
    unittest.main()