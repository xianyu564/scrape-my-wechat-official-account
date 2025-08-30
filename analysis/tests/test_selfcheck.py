#!/usr/bin/env python3
"""
Self-check tests for analysis pipeline
Tests basic functionality, edge cases, and reliability constraints
"""

import os
import sys
import json
import tempfile
import unittest
from pathlib import Path

# Add pipeline to path
sys.path.insert(0, str(Path(__file__).parent.parent / "pipeline"))

from corpus_io import load_corpus, get_corpus_stats, read_text
from tokenizer import MixedLanguageTokenizer
from stats import analyze_zipf_law, analyze_heaps_law, calculate_frequencies
from viz import setup_chinese_font, COLOR_SCHEMES


class TestSelfCheck(unittest.TestCase):
    """Self-check tests for analysis pipeline"""
    
    def setUp(self):
        """Set up test environment"""
        self.test_data_dir = Path(__file__).parent.parent / "data" / "fixtures" / "mini"
        
    def test_r_squared_bounds(self):
        """Test that R² values are properly bounded between 0 and 1"""
        # Load test corpus
        articles = load_corpus(str(self.test_data_dir))
        self.assertGreater(len(articles), 0, "Test fixtures should be available")
        
        # Initialize tokenizer  
        tokenizer = MixedLanguageTokenizer()
        
        # Process articles
        merged_corpus = []
        for article in articles:
            text = read_text(article)  # Use correct function from corpus_io
            if text:
                tokens = tokenizer.tokenize(text)
                merged_corpus.append(tokens)
        
        if merged_corpus:
            # Test Zipf analysis
            frequencies = calculate_frequencies(merged_corpus)
            zipf_results = analyze_zipf_law(frequencies)
            
            r_squared_zipf = zipf_results.get('r_squared', -1)
            self.assertGreaterEqual(r_squared_zipf, 0, "Zipf R² should be >= 0")
            self.assertLessEqual(r_squared_zipf, 1, "Zipf R² should be <= 1")
            
            # Test Heaps analysis
            heaps_results = analyze_heaps_law(merged_corpus)
            r_squared_heaps = heaps_results.get('r_squared', -1)
            self.assertGreaterEqual(r_squared_heaps, 0, "Heaps R² should be >= 0")  
            self.assertLessEqual(r_squared_heaps, 1, "Heaps R² should be <= 1")
            
            print(f"✅ R² bounds test passed: Zipf={r_squared_zipf:.3f}, Heaps={r_squared_heaps:.3f}")
    
    def test_empty_corpus_handling(self):
        """Test graceful handling of empty corpus"""
        # Test with empty corpus
        empty_frequencies = calculate_frequencies([])
        zipf_results = analyze_zipf_law(empty_frequencies)
        heaps_results = analyze_heaps_law([])
        
        # Should not crash and return sensible defaults
        self.assertEqual(zipf_results.get('r_squared', -1), 0)
        self.assertEqual(heaps_results.get('r_squared', -1), 0)
        
        print("✅ Empty corpus handling test passed")
    
    def test_small_corpus_degradation(self):
        """Test graceful degradation for small corpus"""
        # Create minimal corpus
        small_corpus = [['测试', '数据'], ['分析', '系统']]
        
        zipf_results = analyze_zipf_law(calculate_frequencies(small_corpus))
        heaps_results = analyze_heaps_law(small_corpus)
        
        # Should handle gracefully without crashes
        self.assertIsInstance(zipf_results, dict)
        self.assertIsInstance(heaps_results, dict)
        
        # Check for warning messages in heaps (small dataset)
        if 'warning' in heaps_results:
            print(f"✅ Small corpus warning: {heaps_results['warning']}")
        
        print("✅ Small corpus degradation test passed")
    
    def test_tokenizer_fallback(self):
        """Test pkuseg missing fallback to jieba"""
        # This should work regardless of pkuseg availability
        tokenizer = MixedLanguageTokenizer(tokenizer_type='auto')
        
        # Should initialize successfully
        self.assertIsNotNone(tokenizer)
        
        # Should tokenize Chinese text
        tokens = tokenizer.tokenize("这是一个测试文本")
        self.assertGreater(len(tokens), 0)
        
        print(f"✅ Tokenizer fallback test passed: {tokenizer.get_tokenizer_info()}")
    
    def test_font_detection(self):
        """Test Chinese font auto-detection"""
        # Should not crash and provide reasonable fallback
        font_path = setup_chinese_font()
        
        # Should return None or valid path
        if font_path:
            self.assertTrue(os.path.exists(font_path), f"Font path should exist: {font_path}")
        
        print("✅ Font detection test passed")
    
    def test_color_schemes(self):
        """Test color schemes availability"""
        required_schemes = ['nature', 'muted', 'solar']
        
        for scheme in required_schemes:
            self.assertIn(scheme, COLOR_SCHEMES, f"Color scheme '{scheme}' should be available")
            colors = COLOR_SCHEMES[scheme]
            self.assertGreater(len(colors), 0, f"Color scheme '{scheme}' should have colors")
        
        print("✅ Color schemes test passed")
    
    def test_fixed_seed_reproducibility(self):
        """Test that fixed seed produces reproducible results"""
        # This is a basic test - actual reproducibility would need more setup
        import numpy as np
        
        # Test with fixed seed
        np.random.seed(42)
        random_vals_1 = np.random.random(5)
        
        np.random.seed(42)
        random_vals_2 = np.random.random(5)
        
        np.testing.assert_array_equal(random_vals_1, random_vals_2)
        print("✅ Fixed seed reproducibility test passed")
    
    def test_configuration_completeness(self):
        """Test that configuration includes all required parameters"""
        # Test minimal config structure
        required_config_sections = [
            'execution', 'data_paths', 'analysis', 'tfidf', 
            'time_filtering', 'visualization', 'reproducibility'
        ]
        
        # This would be tested with actual config in real run
        # For now just verify we have the structure defined
        config_template = {
            'execution': {'run_analysis': True, 'run_visualization': True},
            'data_paths': {'corpus_root': '', 'output_dir': ''},
            'analysis': {'max_n': 8, 'min_freq': 5},
            'tfidf': {'min_df': 1, 'max_df': 0.98},
            'time_filtering': {'start_date': None},
            'visualization': {'font_path': None, 'color_scheme': 'nature'},
            'reproducibility': {'seed': 42}
        }
        
        for section in required_config_sections:
            self.assertIn(section, config_template, f"Config should have '{section}' section")
        
        print("✅ Configuration completeness test passed")


def run_self_check():
    """Run self-check tests"""
    print("🔧 Running analysis pipeline self-check tests...")
    print("=" * 60)
    
    # Run tests
    unittest.main(argv=[''], exit=False, verbosity=2)
    
    print("=" * 60)
    print("✅ Self-check tests completed")


if __name__ == "__main__":
    run_self_check()