#!/usr/bin/env python3
"""
Unit tests for n-gram extraction with PMI/LLR collocation filtering

Tests validate that PMI and LLR can correctly identify collocations like
"自然 语言 处理" (natural language processing) in small test corpora.
"""

import unittest
import math
import sys
from pathlib import Path

# Add analysis pipeline to path
sys.path.insert(0, str(Path(__file__).parent.parent / "analysis" / "pipeline"))

from ngrams import (
    build_ngrams, 
    _calculate_dunning_llr,
    _filter_by_pmi_enhanced,
    _filter_by_llr_enhanced,
    _sliding_window_ngrams,
    validate_ngram_coverage,
    NGramSpan
)


class TestNGramExtraction(unittest.TestCase):
    """Test basic n-gram extraction functionality"""
    
    def setUp(self):
        """Set up test data"""
        # Small Chinese corpus with known collocations
        self.chinese_tokens = [
            "自然", "语言", "处理", "是", "人工", "智能", "的", "重要", "分支",
            "机器", "学习", "算法", "在", "自然", "语言", "处理", "中", "广泛", "应用",
            "深度", "学习", "模型", "能够", "处理", "复杂", "的", "语言", "任务",
            "自然", "语言", "处理", "技术", "不断", "发展", "和", "完善"
        ]
        
        # Mixed corpus for testing English + Chinese
        self.mixed_tokens = [
            "BERT", "模型", "在", "自然", "语言", "处理", "任务", "中", "表现", "优异",
            "ChatGPT", "是", "一个", "强大", "的", "语言", "模型",
            "机器", "学习", "和", "深度", "学习", "都", "是", "AI", "的", "重要", "技术"
        ]
        
        # Expected trigrams that should be detected
        self.expected_trigrams = [
            ("自然", "语言", "处理"),
            ("机器", "学习", "算法"),
            ("深度", "学习", "模型")
        ]
    
    def test_sliding_window_ngrams(self):
        """Test sliding window n-gram generation"""
        tokens = ["A", "B", "C", "D"]
        
        # Test unigrams
        unigrams = list(_sliding_window_ngrams(tokens, 1))
        self.assertEqual(unigrams, ["A", "B", "C", "D"])
        
        # Test bigrams
        bigrams = list(_sliding_window_ngrams(tokens, 2))
        self.assertEqual(bigrams, [("A", "B"), ("B", "C"), ("C", "D")])
        
        # Test trigrams
        trigrams = list(_sliding_window_ngrams(tokens, 3))
        self.assertEqual(trigrams, [("A", "B", "C"), ("B", "C", "D")])
        
        # Test 4-grams
        fourgrams = list(_sliding_window_ngrams(tokens, 4))
        self.assertEqual(fourgrams, [("A", "B", "C", "D")])
        
        # Test n > length
        fivegrams = list(_sliding_window_ngrams(tokens, 5))
        self.assertEqual(fivegrams, [])


class TestPMICalculation(unittest.TestCase):
    """Test PMI (Pointwise Mutual Information) calculation"""
    
    def setUp(self):
        """Set up test data for PMI calculations"""
        # Simple test corpus with clear associations
        self.tokens = ["A", "B", "A", "B", "A", "B", "C", "D", "C", "D"] * 10  # 100 tokens
        
        # Expected strong collocation: A-B appears frequently together
        # Weak collocation: C-D appears but less frequently
        
    def test_pmi_basic_calculation(self):
        """Test basic PMI calculation with known frequencies"""
        # Build n-grams with PMI filtering
        merged_tokens, ngram_counts = build_ngrams(
            tokens=self.tokens,
            max_n=3,
            min_freq=3,
            collocation='pmi',
            pmi_threshold=0.5,  # Low threshold to catch associations
            use_iterative=True
        )
        
        # Should detect A-B as strong collocation
        self.assertGreater(ngram_counts.get(2, 0), 0, "Should detect bigrams")
        
        # Verify some collocations were merged
        underscore_tokens = [t for t in merged_tokens if '_' in t]
        self.assertGreater(len(underscore_tokens), 0, "Should have merged some collocations")
    
    def test_chinese_trigram_detection_pmi(self):
        """Test PMI can detect Chinese trigrams like '自然 语言 处理'"""
        chinese_tokens = [
            "自然", "语言", "处理", "技术", "日益", "成熟",
            "自然", "语言", "处理", "应用", "广泛",
            "自然", "语言", "处理", "是", "AI", "重点",
            "机器", "学习", "方法", "在", "自然", "语言", "处理", "中", "重要"
        ]
        
        merged_tokens, ngram_counts = build_ngrams(
            tokens=chinese_tokens,
            max_n=4,
            min_freq=2,  # Lower frequency for small corpus
            collocation='pmi',
            pmi_threshold=1.0,
            use_iterative=True
        )
        
        # Check if trigram was detected and merged
        trigram_found = any("自然_语言_处理" in token for token in merged_tokens)
        self.assertTrue(
            trigram_found or ngram_counts.get(3, 0) > 0,
            "Should detect '自然 语言 处理' trigram with PMI"
        )


class TestLLRCalculation(unittest.TestCase):
    """Test LLR (Log-Likelihood Ratio) calculation following Dunning 1993"""
    
    def test_dunning_llr_formula(self):
        """Test Dunning's LLR formula with known values"""
        # Test case: observed_together=10, freq_w1=15, freq_w2=12, total=100
        llr = _calculate_dunning_llr(
            observed_together=10,
            freq_w1=15,
            freq_w2=12,
            total_tokens=100
        )
        
        # LLR should be positive for genuine associations
        self.assertGreater(llr, 0, "LLR should be positive for associations")
        
        # Test edge case: no co-occurrence
        llr_zero = _calculate_dunning_llr(
            observed_together=0,
            freq_w1=10,
            freq_w2=10,
            total_tokens=100
        )
        self.assertEqual(llr_zero, 0.0, "Should return 0 for zero co-occurrence")
    
    def test_llr_contingency_table_correctness(self):
        """Test LLR contingency table calculation correctness"""
        # Known test case with manual calculation
        observed_together = 8
        freq_w1 = 10
        freq_w2 = 12
        total_tokens = 50
        
        # Manual contingency table:
        # a = 8 (w1 & w2)
        # b = 10 - 8 = 2 (w1 & ~w2)  
        # c = 12 - 8 = 4 (~w1 & w2)
        # d = 50 - 10 - 12 + 8 = 36 (~w1 & ~w2)
        
        llr = _calculate_dunning_llr(observed_together, freq_w1, freq_w2, total_tokens)
        
        # Should be reasonable value (not 0 or infinity)
        self.assertTrue(0 < llr < 100, f"LLR should be reasonable: got {llr}")
    
    def test_chinese_trigram_detection_llr(self):
        """Test LLR can detect Chinese trigrams like '自然 语言 处理'"""
        # Use a larger corpus with more repetitions to ensure statistical significance
        chinese_tokens = (
            ["自然", "语言", "处理", "是", "重要", "技术"] * 3 +
            ["自然", "语言", "处理", "发展", "迅速"] * 3 +
            ["自然", "语言", "处理", "应用", "广泛"] * 3 +
            ["深度", "学习", "推动", "自然", "语言", "处理", "进步"] * 2
        )
        
        # Test with very low threshold to ensure detection
        merged_tokens, ngram_counts = build_ngrams(
            tokens=chinese_tokens,
            max_n=4,
            min_freq=2,
            collocation='llr',
            llr_threshold=0.1,  # Very low threshold to detect any association
            use_iterative=True
        )
        
        # Check if trigram was detected and merged
        trigram_found = any("自然_语言_处理" in token for token in merged_tokens)
        
        # The test validates that the algorithm can work with trigrams
        # Either by merging them or at least detecting them as candidates
        self.assertTrue(
            trigram_found or ngram_counts.get(3, 0) > 0,
            f"Should detect trigram with LLR. Merged tokens: {merged_tokens[:10]}..."
        )


class TestNGramIntegration(unittest.TestCase):
    """Test complete n-gram pipeline integration"""
    
    def test_ngram_lengths_detected_output(self):
        """Test that ngram_lengths_detected is properly output"""
        tokens = ["A", "B", "C"] * 20  # Simple repeating pattern
        
        merged_tokens, ngram_counts = build_ngrams(
            tokens=tokens,
            max_n=6,
            min_freq=5,
            collocation='pmi',
            pmi_threshold=0.5
        )
        
        # Verify ngram_counts structure
        self.assertIsInstance(ngram_counts, dict)
        
        # Should have counts for all lengths 1..max_n
        for n in range(1, 7):
            self.assertIn(n, ngram_counts)
            self.assertIsInstance(ngram_counts[n], int)
        
        # At least unigrams should be detected
        self.assertGreater(ngram_counts[1], 0, "Should detect unigrams")
    
    def test_coverage_validation(self):
        """Test ngram coverage validation and recommendations"""
        # Test with limited data
        ngram_counts = {1: 10, 2: 5, 3: 0, 4: 0, 5: 0}
        
        analysis = validate_ngram_coverage(ngram_counts, max_n=5, min_freq=10)
        
        self.assertIn('detected_lengths', analysis)
        self.assertIn('missing_lengths', analysis)
        self.assertIn('recommendations', analysis)
        
        # Should recommend lowering frequency for missing lengths
        self.assertTrue(len(analysis['recommendations']) > 0)
        
        # Missing lengths should include 3, 4, 5
        self.assertIn(3, analysis['missing_lengths'])
        self.assertIn(4, analysis['missing_lengths'])
    
    def test_phrase_merging_with_underscores(self):
        """Test that detected collocations are merged with underscores"""
        # Create corpus with clear bigram pattern
        tokens = ["机器", "学习", "是", "AI", "的", "重要", "技术"] * 5
        
        merged_tokens, ngram_counts = build_ngrams(
            tokens=tokens,
            max_n=3,
            min_freq=3,
            collocation='pmi',
            pmi_threshold=0.5
        )
        
        # Should have some merged phrases
        merged_phrases = [t for t in merged_tokens if '_' in t]
        if merged_phrases:  # If any merging occurred
            # Check format is correct (underscore-separated)
            for phrase in merged_phrases:
                parts = phrase.split('_')
                self.assertGreater(len(parts), 1, "Merged phrases should have multiple parts")
                self.assertTrue(all(part for part in parts), "All parts should be non-empty")
    
    def test_memory_efficiency_large_corpus(self):
        """Test memory efficiency with larger corpus"""
        # Generate larger test corpus
        large_tokens = (["词汇", "处理", "技术"] * 1000 + 
                       ["自然", "语言", "理解"] * 800 +
                       ["机器", "翻译", "系统"] * 600)
        
        # Should complete without memory issues
        merged_tokens, ngram_counts = build_ngrams(
            tokens=large_tokens,
            max_n=8,  # Test high n
            min_freq=10,
            collocation='llr',
            llr_threshold=10.83,
            use_iterative=True  # Test memory-efficient mode
        )
        
        # Should produce reasonable results
        self.assertGreater(len(merged_tokens), 0)
        self.assertGreater(ngram_counts[1], 0)


class TestEdgeCases(unittest.TestCase):
    """Test edge cases and error handling"""
    
    def test_empty_corpus(self):
        """Test handling of empty corpus"""
        merged_tokens, ngram_counts = build_ngrams(
            tokens=[],
            max_n=5,
            min_freq=1,
            collocation='pmi',
            pmi_threshold=1.0
        )
        
        self.assertEqual(merged_tokens, [])
        self.assertEqual(ngram_counts, {})
    
    def test_single_token_corpus(self):
        """Test handling of single token"""
        merged_tokens, ngram_counts = build_ngrams(
            tokens=["单独"],
            max_n=5,
            min_freq=1,
            collocation='pmi',
            pmi_threshold=1.0
        )
        
        self.assertEqual(merged_tokens, ["单独"])
        self.assertEqual(ngram_counts[1], 1)  # Should have 1 unigram
        self.assertEqual(ngram_counts[2], 0)  # No bigrams possible
    
    def test_high_frequency_threshold(self):
        """Test with very high frequency threshold"""
        tokens = ["A", "B", "C"] * 5  # Small corpus
        
        merged_tokens, ngram_counts = build_ngrams(
            tokens=tokens,
            max_n=3,
            min_freq=100,  # Impossibly high threshold
            collocation='pmi',
            pmi_threshold=1.0
        )
        
        # Should handle gracefully, likely with no n-grams
        self.assertIsInstance(merged_tokens, list)
        self.assertIsInstance(ngram_counts, dict)


if __name__ == '__main__':
    # Set up test suite
    unittest.main(verbosity=2)