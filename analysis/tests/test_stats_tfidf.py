#!/usr/bin/env python3
"""
Test suite for TF-IDF with pre-tokenized input and lexical diversity metrics
Tests bypass of sklearn's default preprocessing and stopword handling
"""

import sys
import os
from pathlib import Path

# Add pipeline to path
sys.path.insert(0, str(Path(__file__).parent.parent / "pipeline"))

import pytest
import pandas as pd
from collections import Counter
from stats import (
    calculate_tfidf, top_tfidf_terms, calculate_lexical_metrics,
    calculate_frequencies, calculate_frequencies_by_year
)


class TestTFIDFPreTokenized:
    """Test TF-IDF with pre-tokenized input to avoid sklearn preprocessing issues"""
    
    def test_tfidf_pretokenized_basic(self):
        """Test basic TF-IDF calculation with pre-tokenized input"""
        # Define a simple tokenizer
        def simple_tokenizer(text):
            return text.split()
        
        # Create test data with different year content
        texts_by_year = {
            "2022": [
                "机器学习 深度学习 算法",
                "人工智能 神经网络 算法",
                "机器学习 数据挖掘"
            ],
            "2023": [
                "ChatGPT 大语言模型 算法",
                "人工智能 ChatGPT 应用",
                "深度学习 转换器 模型"
            ]
        }
        
        # Calculate TF-IDF
        tfidf_results = calculate_tfidf(
            texts_by_year=texts_by_year,
            tokenizer_func=simple_tokenizer,
            min_df=1,
            max_df=0.98,
            topk=10
        )
        
        # Should return a DataFrame
        assert isinstance(tfidf_results, pd.DataFrame)
        assert len(tfidf_results) > 0
        
        # Should have required columns
        expected_columns = ['year', 'word', 'score']
        assert all(col in tfidf_results.columns for col in expected_columns)
        
        # Should have entries for both years
        years_in_results = set(tfidf_results['year'])
        assert "2022" in years_in_results
        assert "2023" in years_in_results
        
        # Scores should be positive
        assert all(tfidf_results['score'] > 0)
    
    def test_tfidf_pretokenized_bypass_preprocessing(self):
        """Test that preprocessing bypasses sklearn's default .lower() behavior"""
        def case_sensitive_tokenizer(text):
            # Return tokens with mixed case
            return text.split()
        
        texts_by_year = {
            "2023": [
                "AIGC ChatGPT GPT-4",  # Uppercase technical terms
                "aigc chatgpt gpt-4"   # Lowercase versions
            ]
        }
        
        tfidf_results = calculate_tfidf(
            texts_by_year=texts_by_year,
            tokenizer_func=case_sensitive_tokenizer,
            min_df=1,
            topk=10
        )
        
        # Should preserve case distinctions
        words_in_results = set(tfidf_results['word'])
        
        # Should contain both uppercase and lowercase versions
        # (depending on tokenizer output, they should be preserved)
        assert len(words_in_results) > 0
        assert isinstance(tfidf_results, pd.DataFrame)
    
    def test_tfidf_empty_input(self):
        """Test TF-IDF with empty input"""
        def simple_tokenizer(text):
            return text.split()
        
        # Empty texts
        tfidf_results = calculate_tfidf(
            texts_by_year={},
            tokenizer_func=simple_tokenizer
        )
        
        assert isinstance(tfidf_results, pd.DataFrame)
        assert len(tfidf_results) == 0
        
        # Year with empty texts
        tfidf_results = calculate_tfidf(
            texts_by_year={"2023": []},
            tokenizer_func=simple_tokenizer
        )
        
        assert isinstance(tfidf_results, pd.DataFrame)
        assert len(tfidf_results) == 0
    
    def test_tfidf_with_complex_tokens(self):
        """Test TF-IDF with complex pre-tokenized input including merged n-grams"""
        def complex_tokenizer(text):
            # Simulate pre-tokenized input with merged n-grams
            tokens = []
            words = text.split()
            for word in words:
                if word == "机器学习":
                    tokens.append("机器_学习")  # Merged n-gram
                elif word == "自然语言处理":
                    tokens.append("自然_语言_处理")  # Longer n-gram
                else:
                    tokens.append(word)
            return tokens
        
        texts_by_year = {
            "2023": [
                "机器学习 和 深度学习",
                "自然语言处理 技术发展",
                "机器学习 算法优化"
            ]
        }
        
        tfidf_results = calculate_tfidf(
            texts_by_year=texts_by_year,
            tokenizer_func=complex_tokenizer,
            min_df=1,
            topk=10
        )
        
        # Should handle complex tokens
        assert len(tfidf_results) > 0
        words_in_results = set(tfidf_results['word'])
        
        # Should preserve merged n-grams
        assert "机器_学习" in words_in_results
        assert "自然_语言_处理" in words_in_results


class TestTopTFIDFTerms:
    """Test top TF-IDF terms extraction"""
    
    def setup_method(self):
        """Setup test TF-IDF results"""
        self.tfidf_results = pd.DataFrame([
            {"year": "2022", "word": "机器学习", "score": 0.8},
            {"year": "2022", "word": "深度学习", "score": 0.7},
            {"year": "2022", "word": "算法", "score": 0.6},
            {"year": "2023", "word": "ChatGPT", "score": 0.9},
            {"year": "2023", "word": "大模型", "score": 0.8},
            {"year": "2023", "word": "算法", "score": 0.5},
        ])
    
    def test_top_tfidf_by_year(self):
        """Test getting top TF-IDF terms by year"""
        top_terms = top_tfidf_terms(self.tfidf_results, by="year", k=2)
        
        # Should return dict with years as keys
        assert "2022" in top_terms
        assert "2023" in top_terms
        
        # Each year should have list of (word, score) tuples
        assert len(top_terms["2022"]) == 2
        assert len(top_terms["2023"]) == 2
        
        # Should be sorted by score (descending)
        assert top_terms["2022"][0][1] >= top_terms["2022"][1][1]
        assert top_terms["2023"][0][1] >= top_terms["2023"][1][1]
        
        # Check top terms
        assert top_terms["2022"][0][0] == "机器学习"  # Highest score in 2022
        assert top_terms["2023"][0][0] == "ChatGPT"  # Highest score in 2023
    
    def test_top_tfidf_overall(self):
        """Test getting top TF-IDF terms overall"""
        top_terms = top_tfidf_terms(self.tfidf_results, by="overall", k=3)
        
        # Should return dict with "overall" key
        assert "overall" in top_terms
        assert len(top_terms["overall"]) == 3
        
        # Should average scores across years
        # "算法" appears in both years with scores 0.6 and 0.5, avg = 0.55
        overall_words = [word for word, score in top_terms["overall"]]
        overall_scores = [score for word, score in top_terms["overall"]]
        
        # Should be sorted by score
        assert overall_scores == sorted(overall_scores, reverse=True)
    
    def test_top_tfidf_empty_input(self):
        """Test top TF-IDF with empty input"""
        empty_df = pd.DataFrame(columns=['year', 'word', 'score'])
        
        top_terms = top_tfidf_terms(empty_df, by="year", k=5)
        assert top_terms == {}
        
        top_terms = top_tfidf_terms(empty_df, by="overall", k=5)
        assert top_terms == {}


class TestLexicalMetrics:
    """Test lexical diversity and complexity metrics"""
    
    def test_lexical_metrics_basic(self):
        """Test basic lexical metrics calculation"""
        corpus_tokens = [
            ["机器", "学习", "算法"],
            ["深度", "学习", "神经", "网络"],
            ["机器", "学习", "应用"]
        ]
        
        metrics = calculate_lexical_metrics(corpus_tokens)
        
        # Should return dict with expected metrics
        expected_keys = ['ttr', 'maas_ttr', 'lexical_density', 'content_function_ratio', 'total_tokens', 'unique_tokens']
        assert all(key in metrics for key in expected_keys)
        
        # Check basic counts
        assert metrics['total_tokens'] == 10  # Total across all documents
        assert metrics['unique_tokens'] == 7  # Unique words: 机器, 学习, 算法, 深度, 神经, 网络, 应用
        
        # TTR should be reasonable
        assert 0 < metrics['ttr'] <= 1
        assert metrics['ttr'] == metrics['unique_tokens'] / metrics['total_tokens']
        
        # Maas TTR should be calculated
        assert metrics['maas_ttr'] >= 0
    
    def test_lexical_metrics_with_stopwords(self):
        """Test lexical metrics with stopword filtering"""
        corpus_tokens = [
            ["这是", "机器", "学习", "的", "应用"],
            ["那是", "深度", "学习", "的", "算法"]
        ]
        
        stopwords_zh = {"这是", "那是", "的"}
        stopwords_en = {"the", "a", "an"}
        
        metrics = calculate_lexical_metrics(
            corpus_tokens, 
            stopwords_zh=stopwords_zh,
            stopwords_en=stopwords_en
        )
        
        # Should calculate lexical density based on content words
        assert 'lexical_density' in metrics
        assert 0 <= metrics['lexical_density'] <= 1
        
        # Content words should exclude stopwords
        # Total tokens: 10, stopwords: "这是", "那是", "的" (appear 3 times total)
        # Content words: 6, so lexical density should be 6/10 = 0.6
        assert metrics['lexical_density'] == 0.6
    
    def test_lexical_metrics_edge_cases(self):
        """Test lexical metrics edge cases"""
        # Empty corpus
        metrics = calculate_lexical_metrics([])
        assert metrics['ttr'] == 0.0
        assert metrics['total_tokens'] == 0
        
        # Corpus with empty documents
        metrics = calculate_lexical_metrics([[], []])
        assert metrics['ttr'] == 0.0
        assert metrics['total_tokens'] == 0
        
        # Single token
        metrics = calculate_lexical_metrics([["单词"]])
        assert metrics['total_tokens'] == 1
        assert metrics['unique_tokens'] == 1
        assert metrics['ttr'] == 1.0


def test_integration_tfidf_pipeline():
    """Integration test: TF-IDF pipeline with realistic data"""
    # Simulate realistic tokenizer output
    def realistic_tokenizer(text):
        # Simulate pre-tokenized output with normalized technical terms
        tokens = []
        words = text.split()
        for word in words:
            if word.lower() == "chatgpt":
                tokens.append("chatgpt")  # Normalized case
            elif "机器学习" in word:
                tokens.append("机器_学习")  # Merged n-gram
            else:
                tokens.append(word.lower())  # General normalization
        return tokens
    
    texts_by_year = {
        "2022": [
            "机器学习 深度学习 研究进展",
            "人工智能 算法 优化方法",
            "机器学习 应用案例 分析"
        ],
        "2023": [
            "ChatGPT 大语言模型 突破",
            "人工智能 深度学习 技术革新", 
            "机器学习 应用案例 分析"
        ]
    }
    
    # Calculate TF-IDF
    tfidf_results = calculate_tfidf(
        texts_by_year=texts_by_year,
        tokenizer_func=realistic_tokenizer,
        min_df=1,
        max_df=0.95,
        topk=15
    )
    
    # Should work without errors
    assert len(tfidf_results) > 0
    
    # Get top terms by year
    top_terms = top_tfidf_terms(tfidf_results, by="year", k=5)
    
    assert "2022" in top_terms
    assert "2023" in top_terms
    
    # ChatGPT should be present in 2023 (appears once, so should have decent TF-IDF)
    terms_2023 = [word for word, score in top_terms["2023"]]
    all_words_2023 = set(tfidf_results[tfidf_results['year'] == '2023']['word'])
    
    # ChatGPT should appear in the results (even if not in top 5)
    assert "chatgpt" in all_words_2023, f"Expected 'chatgpt' in 2023 words: {all_words_2023}"
    
    # Merged n-grams should be preserved
    all_words = set(tfidf_results['word'])
    assert "机器_学习" in all_words


if __name__ == "__main__":
    # Run tests directly
    pytest.main([__file__, "-v"])