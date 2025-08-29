"""Unit tests for freq_stats module."""

import unittest
import tempfile
import os
import pandas as pd
import numpy as np
from pathlib import Path
import sys

# Add the source path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from freq_stats import (
    term_freq_overall, term_freq_by_year, zipf_plot,
    save_freq_stats, get_stats_summary
)


class TestFreqStats(unittest.TestCase):
    """Test frequency statistics functionality."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        
        # Sample corpus tokens
        self.corpus_tokens = [
            ['机器', '学习', '算法', '数据'],
            ['深度', '学习', '神经', '网络'],
            ['自然', '语言', '处理', '算法'],
            ['机器', '学习', '应用', '实践']
        ]
        
        # Sample corpus by year
        self.corpus_by_year = {
            '2022': [
                ['机器', '学习', '算法'],
                ['深度', '学习', '网络']
            ],
            '2023': [
                ['自然', '语言', '处理'],
                ['机器', '学习', '应用']
            ]
        }
    
    def tearDown(self):
        """Clean up test fixtures."""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_term_freq_overall(self):
        """Test overall term frequency calculation."""
        freq_df = term_freq_overall(self.corpus_tokens)
        
        # Should return a DataFrame
        self.assertIsInstance(freq_df, pd.DataFrame)
        
        # Should have correct columns
        self.assertListEqual(list(freq_df.columns), ['word', 'freq'])
        
        # Should be sorted by frequency (descending)
        self.assertTrue(freq_df['freq'].is_monotonic_decreasing)
        
        # Check specific frequencies
        word_freq_dict = dict(zip(freq_df['word'], freq_df['freq']))
        self.assertEqual(word_freq_dict.get('学习', 0), 3)  # Appears 3 times
        self.assertEqual(word_freq_dict.get('算法', 0), 2)  # Appears 2 times
        self.assertEqual(word_freq_dict.get('机器', 0), 2)  # Appears 2 times
    
    def test_term_freq_by_year(self):
        """Test term frequency by year calculation."""
        freq_df = term_freq_by_year(self.corpus_by_year)
        
        # Should return a DataFrame
        self.assertIsInstance(freq_df, pd.DataFrame)
        
        # Should have correct columns
        self.assertListEqual(list(freq_df.columns), ['year', 'word', 'freq'])
        
        # Should contain expected years
        years = set(freq_df['year'].unique())
        self.assertEqual(years, {'2022', '2023'})
        
        # Check specific year-word frequencies
        year_2022_data = freq_df[freq_df['year'] == '2022']
        word_freq_2022 = dict(zip(year_2022_data['word'], year_2022_data['freq']))
        self.assertEqual(word_freq_2022.get('学习', 0), 2)  # Appears 2 times in 2022
    
    def test_empty_corpus(self):
        """Test handling of empty corpus."""
        # Empty overall corpus
        freq_df = term_freq_overall([])
        self.assertTrue(freq_df.empty)
        
        # Empty year corpus
        freq_df = term_freq_by_year({})
        self.assertTrue(freq_df.empty)
    
    def test_zipf_plot_creation(self):
        """Test Zipf plot creation."""
        freq_df = term_freq_overall(self.corpus_tokens)
        output_path = os.path.join(self.temp_dir, 'test_zipf.png')
        
        # Should not raise an exception
        try:
            zipf_plot(freq_df, output_path, title="Test Zipf Plot")
            plot_created = True
        except Exception:
            plot_created = False
        
        # Note: We can't easily test the actual plot content,
        # but we can test that it doesn't crash
        self.assertTrue(plot_created)
    
    def test_save_freq_stats(self):
        """Test saving frequency statistics to CSV."""
        freq_overall = term_freq_overall(self.corpus_tokens)
        freq_by_year = term_freq_by_year(self.corpus_by_year)
        
        # Create a dummy TF-IDF DataFrame
        tfidf_by_year = pd.DataFrame({
            'year': ['2022', '2022', '2023', '2023'],
            'word': ['机器', '学习', '自然', '语言'],
            'score': [0.8, 0.7, 0.9, 0.6]
        })
        
        save_freq_stats(freq_overall, freq_by_year, tfidf_by_year, self.temp_dir)
        
        # Check that files were created
        self.assertTrue(os.path.exists(os.path.join(self.temp_dir, 'freq_overall.csv')))
        self.assertTrue(os.path.exists(os.path.join(self.temp_dir, 'freq_by_year.csv')))
        self.assertTrue(os.path.exists(os.path.join(self.temp_dir, 'tfidf_topk_by_year.csv')))
        
        # Verify file contents
        loaded_overall = pd.read_csv(os.path.join(self.temp_dir, 'freq_overall.csv'))
        self.assertEqual(len(loaded_overall), len(freq_overall))
        self.assertListEqual(list(loaded_overall.columns), ['word', 'freq'])
    
    def test_get_stats_summary(self):
        """Test getting statistics summary."""
        freq_overall = term_freq_overall(self.corpus_tokens)
        freq_by_year = term_freq_by_year(self.corpus_by_year)
        
        # Create a dummy TF-IDF DataFrame
        tfidf_by_year = pd.DataFrame({
            'year': ['2022', '2022', '2023', '2023'],
            'word': ['机器', '学习', '自然', '语言'],
            'score': [0.8, 0.7, 0.9, 0.6]
        })
        
        summary = get_stats_summary(freq_overall, freq_by_year, tfidf_by_year)
        
        # Should be a dictionary
        self.assertIsInstance(summary, dict)
        
        # Should contain expected keys
        expected_keys = [
            'total_unique_words', 'total_word_freq', 'top_words',
            'years', 'words_by_year', 'freq_by_year',
            'tfidf_years', 'top_tfidf_words'
        ]
        
        for key in expected_keys:
            self.assertIn(key, summary)
        
        # Check some values
        self.assertEqual(summary['total_unique_words'], len(freq_overall))
        self.assertEqual(summary['total_word_freq'], freq_overall['freq'].sum())
        self.assertEqual(set(summary['years']), {'2022', '2023'})
    
    def test_edge_cases(self):
        """Test edge cases."""
        # Single word corpus
        single_word_corpus = [['测试']]
        freq_df = term_freq_overall(single_word_corpus)
        self.assertEqual(len(freq_df), 1)
        self.assertEqual(freq_df.iloc[0]['word'], '测试')
        self.assertEqual(freq_df.iloc[0]['freq'], 1)
        
        # Corpus with repeated tokens
        repeated_corpus = [['重复', '重复', '重复']]
        freq_df = term_freq_overall(repeated_corpus)
        word_freq_dict = dict(zip(freq_df['word'], freq_df['freq']))
        self.assertEqual(word_freq_dict['重复'], 3)


if __name__ == '__main__':
    # Skip matplotlib-dependent tests in headless environment
    import matplotlib
    matplotlib.use('Agg')  # Use non-GUI backend
    
    unittest.main()