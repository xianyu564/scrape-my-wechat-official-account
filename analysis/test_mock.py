#!/usr/bin/env python3
"""
Mock test for core functionality without external dependencies
"""

import sys
from pathlib import Path
from collections import Counter

# Add pipeline to path
sys.path.insert(0, str(Path(__file__).parent / "pipeline"))

def test_ngram_functionality():
    """Test n-gram functionality with mock data"""
    print("🔢 Testing n-gram functionality...")
    
    try:
        from ngrams import _extract_ngrams, _filter_by_pmi, get_ngram_stats
        
        # Test data
        tokens = ["机器", "学习", "是", "人工", "智能", "的", "重要", "分支"]
        
        # Test unigrams
        unigrams = _extract_ngrams(tokens, 1)
        assert len(unigrams) == len(set(tokens)), "Unigram count mismatch"
        print("✅ Unigram extraction works")
        
        # Test bigrams
        bigrams = _extract_ngrams(tokens, 2)
        expected_bigrams = len(tokens) - 1
        assert len(bigrams) == expected_bigrams, f"Expected {expected_bigrams} bigrams, got {len(bigrams)}"
        print("✅ Bigram extraction works")
        
        # Test n-gram stats
        mock_tokens = ["机器_学习", "人工_智能", "深度", "学习", "neural_network"]
        stats = get_ngram_stats(mock_tokens)
        assert stats['compound_ngrams'] == 2, "Compound n-gram count incorrect"
        assert stats['total_tokens'] == 5, "Total token count incorrect"
        print("✅ N-gram statistics work")
        
        return True
        
    except Exception as e:
        print(f"❌ N-gram test failed: {e}")
        return False


def test_corpus_io():
    """Test corpus I/O functionality"""
    print("\n📁 Testing corpus I/O...")
    
    try:
        from corpus_io import Article, get_corpus_stats
        
        # Create mock articles
        articles = [
            Article("2023-01-01", "2023", "Test 1", "", "test1.md", ""),
            Article("2023-06-01", "2023", "Test 2", "", "test2.md", ""),
            Article("2024-01-01", "2024", "Test 3", "", "test3.md", "")
        ]
        
        stats = get_corpus_stats(articles)
        
        assert stats['total_articles'] == 3, "Article count incorrect"
        assert len(stats['years']) == 2, "Year count incorrect"
        assert '2023' in stats['years'], "Year 2023 missing"
        assert '2024' in stats['years'], "Year 2024 missing"
        
        print("✅ Corpus statistics work")
        return True
        
    except Exception as e:
        print(f"❌ Corpus I/O test failed: {e}")
        return False


def test_normalizer():
    """Test text normalizer"""
    print("\n🔤 Testing text normalizer...")
    
    try:
        from tokenize import ChineseNormalizer
        
        normalizer = ChineseNormalizer()
        
        # Test full-width to half-width conversion
        text = "，。！？（）"
        normalized = normalizer.normalize(text)
        expected = ",.!?()"
        assert normalized == expected, f"Expected '{expected}', got '{normalized}'"
        
        # Test English lowercasing
        text = "Machine-Learning AIGC_2025"
        normalized = normalizer.normalize(text)
        assert "machine-learning" in normalized, "English not properly normalized"
        assert "aigc_2025" in normalized, "Technical term not properly normalized"
        
        print("✅ Text normalization works")
        return True
        
    except Exception as e:
        print(f"❌ Normalizer test failed: {e}")
        return False


def test_frequency_analysis():
    """Test frequency analysis without external dependencies"""
    print("\n📊 Testing frequency analysis...")
    
    try:
        from stats import calculate_frequencies, analyze_zipf_law, calculate_lexical_metrics
        
        # Mock corpus
        corpus = [
            ["机器", "学习", "很", "重要"],
            ["人工", "智能", "是", "未来"],
            ["机器", "学习", "和", "深度", "学习"]
        ]
        
        # Test frequency calculation
        frequencies = calculate_frequencies(corpus)
        assert frequencies["机器"] == 2, "Frequency count incorrect"
        assert frequencies["学习"] == 3, "Frequency count incorrect"
        
        # Test Zipf analysis (should handle small data gracefully)
        zipf_results = analyze_zipf_law(frequencies)
        assert 'slope' in zipf_results, "Zipf results missing slope"
        
        # Test lexical metrics
        metrics = calculate_lexical_metrics(corpus)
        assert metrics['total_tokens'] > 0, "Token count should be positive"
        assert 0 <= metrics['ttr'] <= 1, "TTR should be between 0 and 1"
        
        print("✅ Frequency analysis works")
        return True
        
    except Exception as e:
        print(f"❌ Frequency analysis test failed: {e}")
        return False


def test_two_phase_design():
    """Test that the two-phase design structure is correctly implemented"""
    print("\n🏗️  Testing two-phase design...")
    
    try:
        # Check main.py has the correct functions
        with open("main.py", 'r', encoding='utf-8') as f:
            content = f.read()
        
        required_functions = [
            "def run_analysis(",
            "def run_presentation(",
            "PHASE 1: THEORY-ONLY ANALYSIS",
            "PHASE 2: PRESENTATION"
        ]
        
        for func in required_functions:
            assert func in content, f"Missing: {func}"
        
        print("✅ Two-phase design structure correct")
        return True
        
    except Exception as e:
        print(f"❌ Two-phase design test failed: {e}")
        return False


def main():
    """Run all mock tests"""
    print("=" * 60)
    print("🧪 MOCK FUNCTIONALITY TESTS")
    print("=" * 60)
    
    tests = [
        ("N-gram Functionality", test_ngram_functionality),
        ("Corpus I/O", test_corpus_io),
        ("Text Normalizer", test_normalizer),
        ("Frequency Analysis", test_frequency_analysis),
        ("Two-Phase Design", test_two_phase_design)
    ]
    
    results = []
    
    for test_name, test_func in tests:
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"❌ {test_name} failed with error: {e}")
            results.append((test_name, False))
    
    # Summary
    print("\n" + "=" * 60)
    print("📊 MOCK TEST SUMMARY")
    print("=" * 60)
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for test_name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{status} {test_name}")
    
    print(f"\n🎯 Overall: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All core functionality tests passed!")
        print("📝 Implementation ready for full testing with dependencies")
    else:
        print("⚠️  Some functionality tests failed.")
    
    return passed == total


if __name__ == "__main__":
    main()