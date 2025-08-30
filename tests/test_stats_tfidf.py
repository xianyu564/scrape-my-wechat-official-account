#!/usr/bin/env python3
"""
Test suite for TF-IDF and lexical diversity functionality in stats.py
Ensures pre-tokenized input works correctly without triggering sklearn's .lower() or regex splitting
"""

import pytest
import numpy as np
import pandas as pd
from collections import Counter
import sys
import os
from pathlib import Path

# Add analysis directory to path for imports
ANALYSIS_PATH = Path(__file__).parent.parent / "analysis"
sys.path.insert(0, str(ANALYSIS_PATH))

from pipeline.stats import (
    calculate_tfidf, 
    top_tfidf_terms,
    calculate_lexical_metrics,
    get_year_over_year_growth,
    calculate_frequencies,
    calculate_frequencies_by_year
)


class TestTFIDFPreTokenized:
    """Test TF-IDF implementation with pre-tokenized input"""
    
    def setup_method(self):
        """Set up test data with fixed random seed for reproducibility"""
        np.random.seed(42)
        
        # Test data with mixed case, Chinese characters, and technical terms
        # This tests that .lower() is NOT applied
        self.texts_by_year = {
            '2023': [
                'Machine Learning is Important',
                'Deep Learning and AI',
                'Python Programming Tutorial'
            ],
            '2024': [
                'AIGC_2025 Technology Trends',
                'Machine Learning Advances',
                'Natural Language Processing NLP'
            ]
        }
        
        # Pre-tokenized version preserving case and technical terms
        self.tokenized_by_year = {
            '2023': [
                ['Machine', 'Learning', 'is', 'Important'],
                ['Deep', 'Learning', 'and', 'AI'],
                ['Python', 'Programming', 'Tutorial']
            ],
            '2024': [
                ['AIGC_2025', 'Technology', 'Trends'],
                ['Machine', 'Learning', 'Advances'],
                ['Natural', 'Language', 'Processing', 'NLP']
            ]
        }
        
        # Simple tokenizer that splits on spaces and preserves case
        def case_preserving_tokenizer(text):
            return text.split()
        
        self.tokenizer_func = case_preserving_tokenizer
    
    def test_tfidf_preserves_case(self):
        """Test that TF-IDF does not apply .lower() to pre-tokenized input"""
        result = calculate_tfidf(
            self.texts_by_year, 
            self.tokenizer_func,
            min_df=1,
            max_df=0.98,
            topk=10
        )
        
        # Check that mixed case terms are preserved
        words_in_result = set(result['word'].values)
        
        # These should be present with original case
        assert 'Machine' in words_in_result, "Mixed case 'Machine' should be preserved"
        assert 'AIGC_2025' in words_in_result, "Technical term 'AIGC_2025' should be preserved"
        assert 'NLP' in words_in_result, "Acronym 'NLP' should be preserved"
        
        # These should NOT be present (would indicate .lower() was applied)
        assert 'machine' not in words_in_result, "Lowercase 'machine' should not exist (case preserved)"
        assert 'aigc_2025' not in words_in_result, "Lowercase 'aigc_2025' should not exist"
    
    def test_tfidf_no_regex_splitting(self):
        """Test that technical terms with underscores are not split by regex"""
        result = calculate_tfidf(
            self.texts_by_year,
            self.tokenizer_func,
            min_df=1,
            topk=10
        )
        
        words_in_result = set(result['word'].values)
        
        # Technical term should remain intact
        assert 'AIGC_2025' in words_in_result, "Underscore term should not be split"
        
        # Should not be split into parts
        assert 'AIGC' not in words_in_result, "Should not split AIGC_2025 into AIGC"
        assert '2025' not in words_in_result, "Should not split AIGC_2025 into 2025"
    
    def test_tfidf_output_structure(self):
        """Test that TF-IDF output has correct structure"""
        result = calculate_tfidf(
            self.texts_by_year,
            self.tokenizer_func,
            topk=5
        )
        
        # Check DataFrame structure
        expected_columns = {'year', 'word', 'score'}
        assert set(result.columns) == expected_columns, f"Expected columns {expected_columns}, got {set(result.columns)}"
        
        # Check data types
        assert result['score'].dtype in [np.float64, np.float32], "TF-IDF scores should be float"
        assert result['year'].dtype == object, "Year should be object/string type"
        assert result['word'].dtype == object, "Word should be object/string type"
        
        # Check that all scores are positive
        assert (result['score'] >= 0).all(), "All TF-IDF scores should be non-negative"
    
    def test_top_tfidf_terms_by_year(self):
        """Test top_tfidf_terms with by='year' parameter"""
        tfidf_results = calculate_tfidf(
            self.texts_by_year,
            self.tokenizer_func,
            topk=10
        )
        
        top_terms = top_tfidf_terms(tfidf_results, by="year", k=3)
        
        # Should have results for each year
        assert '2023' in top_terms, "Should have results for 2023"
        assert '2024' in top_terms, "Should have results for 2024"
        
        # Each year should have up to k terms
        for year, terms in top_terms.items():
            assert len(terms) <= 3, f"Should have at most 3 terms for {year}"
            for word, score in terms:
                assert isinstance(word, str), "Word should be string"
                assert isinstance(score, (int, float)), "Score should be numeric"
                assert score >= 0, "Score should be non-negative"
    
    def test_top_tfidf_terms_overall(self):
        """Test top_tfidf_terms with by='overall' parameter"""
        tfidf_results = calculate_tfidf(
            self.texts_by_year,
            self.tokenizer_func,
            topk=10
        )
        
        top_terms = top_tfidf_terms(tfidf_results, by="overall", k=5)
        
        # Should have overall results
        assert "overall" in top_terms, "Should have overall results"
        
        overall_terms = top_terms["overall"]
        assert len(overall_terms) <= 5, "Should have at most 5 overall terms"
        
        for word, score in overall_terms:
            assert isinstance(word, str), "Word should be string"
            assert isinstance(score, (int, float)), "Score should be numeric"
            assert score >= 0, "Score should be non-negative"
    
    def test_reproducibility_with_fixed_seed(self):
        """Test that results are reproducible with fixed random seed"""
        np.random.seed(42)
        result1 = calculate_tfidf(
            self.texts_by_year,
            self.tokenizer_func,
            topk=5
        )
        
        np.random.seed(42)
        result2 = calculate_tfidf(
            self.texts_by_year,
            self.tokenizer_func,
            topk=5
        )
        
        # Results should be identical
        pd.testing.assert_frame_equal(result1, result2, "Results should be reproducible with fixed seed")


class TestLexicalDiversityMetrics:
    """Test lexical diversity metrics implementation"""
    
    def setup_method(self):
        """Set up test data for lexical metrics"""
        # Load stopwords
        self.stopwords_zh = self._load_stopwords('zh')
        self.stopwords_en = self._load_stopwords('en')
        
        # Test corpus with mixed content and function words
        self.test_corpus = [
            ['Machine', 'Learning', 'is', 'a', 'powerful', 'tool'],
            ['The', 'algorithm', 'can', 'process', 'natural', 'language'],
            ['Deep', 'neural', 'networks', 'are', 'very', 'effective']
        ]
    
    def _load_stopwords(self, lang):
        """Load stopwords from data files"""
        stopwords_path = ANALYSIS_PATH / f"data/stopwords.{lang}.txt"
        if stopwords_path.exists():
            with open(stopwords_path, 'r', encoding='utf-8') as f:
                return set(line.strip() for line in f if line.strip())
        return set()
    
    def test_basic_ttr_calculation(self):
        """Test basic Type-Token Ratio calculation"""
        metrics = calculate_lexical_metrics(self.test_corpus)
        
        # Check that TTR is calculated
        assert 'ttr' in metrics, "TTR should be calculated"
        assert 0 <= metrics['ttr'] <= 1, "TTR should be between 0 and 1"
        
        # Check token counts
        total_tokens = sum(len(tokens) for tokens in self.test_corpus)
        unique_tokens = len(set(token for tokens in self.test_corpus for token in tokens))
        
        assert metrics['total_tokens'] == total_tokens, "Total tokens should match"
        assert metrics['unique_tokens'] == unique_tokens, "Unique tokens should match"
        
        expected_ttr = unique_tokens / total_tokens
        assert abs(metrics['ttr'] - expected_ttr) < 1e-6, "TTR calculation should be correct"
    
    def test_maas_ttr_calculation(self):
        """Test Maas TTR calculation"""
        metrics = calculate_lexical_metrics(self.test_corpus)
        
        assert 'maas_ttr' in metrics, "Maas TTR should be calculated"
        assert metrics['maas_ttr'] >= 0, "Maas TTR should be non-negative"
    
    def test_lexical_density_with_stopwords(self):
        """Test lexical density calculation with stopwords"""
        metrics = calculate_lexical_metrics(
            self.test_corpus,
            stopwords_zh=self.stopwords_zh,
            stopwords_en=self.stopwords_en
        )
        
        assert 'lexical_density' in metrics, "Lexical density should be calculated"
        assert 0 <= metrics['lexical_density'] <= 1, "Lexical density should be between 0 and 1"
        
        assert 'content_function_ratio' in metrics, "Content/function ratio should be calculated"
        assert metrics['content_function_ratio'] >= 0, "Content/function ratio should be non-negative"
    
    def test_stopwords_integration(self):
        """Test that stopwords are properly loaded and used"""
        # Test with English stopwords
        assert len(self.stopwords_en) > 0, "English stopwords should be loaded"
        assert 'the' in self.stopwords_en, "Common English stopwords should be present"
        assert 'is' in self.stopwords_en, "Common English stopwords should be present"
        
        # Test with Chinese stopwords  
        assert len(self.stopwords_zh) > 0, "Chinese stopwords should be loaded"
        assert '的' in self.stopwords_zh, "Common Chinese stopwords should be present"
        assert '是' in self.stopwords_zh, "Common Chinese stopwords should be present"


class TestYearOverYearAnalysis:
    """Test year-over-year growth analysis"""
    
    def setup_method(self):
        """Set up test data for YoY analysis"""
        self.freq_by_year = {
            '2022': Counter({'word1': 10, 'word2': 5, 'word3': 3}),
            '2023': Counter({'word1': 15, 'word2': 3, 'word4': 8}),
            '2024': Counter({'word1': 20, 'word3': 10, 'word4': 12})
        }
    
    def test_yoy_growth_calculation(self):
        """Test year-over-year growth calculation"""
        growth_data = get_year_over_year_growth(self.freq_by_year, topk=10)
        
        assert len(growth_data) > 0, "Should calculate growth data"
        
        # Check structure
        for item in growth_data:
            required_keys = {'word', 'prev_year', 'curr_year', 'prev_count', 'curr_count', 'growth'}
            assert set(item.keys()) == required_keys, f"Growth item should have keys {required_keys}"
            
            # Check growth calculation
            expected_growth = item['curr_count'] - item['prev_count']
            assert item['growth'] == expected_growth, "Growth should be calculated correctly"
    
    def test_yoy_with_insufficient_years(self):
        """Test YoY analysis with insufficient years"""
        single_year_freq = {'2023': Counter({'word1': 10})}
        growth_data = get_year_over_year_growth(single_year_freq)
        
        assert growth_data == [], "Should return empty list for single year"


class TestIntegrationTests:
    """Integration tests for end-to-end functionality"""
    
    def test_full_pipeline_with_chinese_text(self):
        """Test full pipeline with Chinese text to ensure no encoding issues"""
        chinese_texts = {
            '2023': ['机器学习很重要', '深度学习和人工智能'],
            '2024': ['自然语言处理技术', '机器学习的进展']
        }
        
        def chinese_tokenizer(text):
            # Simple Chinese tokenizer (in practice would use jieba/pkuseg)
            return list(text)  # Character-level for testing
        
        # Should not crash with Chinese input
        result = calculate_tfidf(chinese_texts, chinese_tokenizer, topk=5)
        
        assert not result.empty, "Should handle Chinese text"
        assert len(result) > 0, "Should produce results for Chinese text"
    
    def test_empty_input_handling(self):
        """Test handling of empty inputs"""
        # Empty texts
        empty_result = calculate_tfidf({}, lambda x: x.split())
        assert empty_result.empty, "Should handle empty input gracefully"
        
        # Empty lexical metrics
        empty_metrics = calculate_lexical_metrics([])
        expected_keys = {'ttr', 'maas_ttr', 'lexical_density', 'content_function_ratio', 'total_tokens', 'unique_tokens'}
        assert set(empty_metrics.keys()) == expected_keys, "Should return default metrics for empty input"
        assert empty_metrics['total_tokens'] == 0, "Total tokens should be 0 for empty input"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])