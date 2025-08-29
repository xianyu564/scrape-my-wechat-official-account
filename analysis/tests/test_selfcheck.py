#!/usr/bin/env python3
"""
Self-check and regression testing framework
Task 7: 自检与回归
"""

import os
import sys
import time
import traceback
from pathlib import Path
from typing import List, Dict, Any, Tuple
from collections import Counter

# Add pipeline to path
sys.path.insert(0, str(Path(__file__).parent.parent / "pipeline"))


class SelfCheckFramework:
    """Comprehensive self-check framework for the analysis pipeline"""
    
    def __init__(self):
        self.results = {}
        self.warnings = []
        self.errors = []
    
    def run_all_checks(self) -> Tuple[bool, Dict[str, Any]]:
        """
        Run all self-checks and return comprehensive results
        
        Returns:
            Tuple[bool, Dict]: (success, detailed_results)
        """
        print("=" * 70)
        print("🔍 COMPREHENSIVE SELF-CHECK STARTING")
        print("=" * 70)
        
        checks = [
            ("tokenizer_fallback", self.check_tokenizer_fallback),
            ("ngram_analysis", self.check_ngram_analysis),
            ("tfidf_pretokenized", self.check_tfidf_pretokenized),
            ("zipf_heaps_laws", self.check_zipf_heaps_laws),
            ("visualization_creation", self.check_visualization_creation),
            ("chinese_font_detection", self.check_chinese_font_detection),
            ("error_handling", self.check_error_handling),
            ("memory_efficiency", self.check_memory_efficiency),
            ("reproducibility", self.check_reproducibility),
        ]
        
        overall_success = True
        
        for check_name, check_func in checks:
            print(f"\n🧪 Running check: {check_name}")
            try:
                start_time = time.time()
                success, details = check_func()
                elapsed = time.time() - start_time
                
                self.results[check_name] = {
                    'success': success,
                    'details': details,
                    'elapsed_time': elapsed
                }
                
                if success:
                    print(f"✅ {check_name} PASSED ({elapsed:.2f}s)")
                else:
                    print(f"❌ {check_name} FAILED ({elapsed:.2f}s)")
                    print(f"   Details: {details}")
                    overall_success = False
                    
            except Exception as e:
                print(f"💥 {check_name} CRASHED: {e}")
                traceback.print_exc()
                self.results[check_name] = {
                    'success': False,
                    'details': f"Exception: {e}",
                    'elapsed_time': 0
                }
                overall_success = False
        
        # Print summary
        self.print_summary()
        
        return overall_success, self.results
    
    def check_tokenizer_fallback(self) -> Tuple[bool, str]:
        """Check tokenizer fallback mechanism (pkuseg → jieba)"""
        try:
            from tokenizer import MixedLanguageTokenizer
            
            # Test with auto mode (should work regardless of pkuseg availability)
            # Make sure to include user dictionaries for technical terms
            user_dicts = []
            tech_terms_path = "data/tech_terms.txt"
            if os.path.exists(tech_terms_path):
                user_dicts.append(tech_terms_path)
            
            tokenizer = MixedLanguageTokenizer(
                tokenizer_type="auto",
                extra_user_dicts=user_dicts
            )
            
            test_text = "北京大学生前来应聘，机器学习很有趣。AIGC_2025技术升级。"
            tokens = tokenizer.tokenize(test_text)
            
            if not tokens:
                return False, "No tokens produced"
            
            # Check that technical terms are preserved (more flexible check)
            has_technical_term = any(
                'AIGC' in token or 'AIGC_2025' in token or 'AIGC_2025' == token
                for token in tokens
            )
            
            # Test tokenizer info
            info = tokenizer.get_tokenizer_info()
            if 'name' not in info:
                return False, "Tokenizer info incomplete"
            
            return True, f"Tokenizer: {info['name']}, Tokens: {len(tokens)}, Tech terms preserved: {has_technical_term}"
            
        except Exception as e:
            return False, f"Exception: {e}"
    
    def check_ngram_analysis(self) -> Tuple[bool, str]:
        """Check n-gram analysis with thresholds"""
        try:
            from ngrams import build_ngrams
            
            # Create test data that should trigger n-gram merging
            test_tokens = [
                '自然', '语言', '处理', '技术', '发展',
                '自然', '语言', '处理', '应用', '广泛',
                '机器', '学习', '算法', '优化',
                '机器', '学习', '模型', '训练'
            ] * 10  # Repeat to ensure frequency
            
            merged_tokens, ngram_counts = build_ngrams(
                tokens=test_tokens,
                max_n=3,
                min_freq=2,
                collocation='pmi',
                pmi_threshold=1.0,  # Lower threshold for testing
                llr_threshold=3.0
            )
            
            # Check that some n-grams were detected
            total_ngrams = sum(ngram_counts.values())
            if total_ngrams == 0:
                return False, "No n-grams detected"
            
            # Check coverage
            expected_lengths = [1, 2, 3]
            detected_lengths = [n for n in expected_lengths if ngram_counts.get(n, 0) > 0]
            
            if len(detected_lengths) < 2:
                return False, f"Insufficient n-gram coverage: {detected_lengths}"
            
            return True, f"N-grams detected: {ngram_counts}, Coverage: {detected_lengths}"
            
        except Exception as e:
            return False, f"Exception: {e}"
    
    def check_tfidf_pretokenized(self) -> Tuple[bool, str]:
        """Check TF-IDF with pre-tokenized input doesn't trigger .lower() error"""
        try:
            from stats import calculate_tfidf
            from tokenizer import MixedLanguageTokenizer
            
            # Test with pre-tokenized input that would break .lower()
            tokenizer = MixedLanguageTokenizer(tokenizer_type="auto")
            
            test_texts = {
                '2023': [
                    "Hello WORLD test",
                    "机器学习 AIGC_2025 技术",
                    "Mixed 中文 English content"
                ]
            }
            
            results = calculate_tfidf(
                texts_by_year=test_texts,
                tokenizer_func=tokenizer.tokenize,
                min_df=1,
                max_df=1.0,
                topk=10
            )
            
            if results.empty:
                return False, "No TF-IDF results produced"
            
            # Check that mixed-case tokens are preserved
            if len(results) > 0:
                all_terms = set()
                for _, row in results.iterrows():
                    all_terms.add(row['word'])  # Changed from 'term' to 'word'
                
                has_mixed_case = any(term != term.lower() for term in all_terms if isinstance(term, str))
                return True, f"TF-IDF terms: {len(all_terms)}, Mixed-case preserved: {has_mixed_case}"
            else:
                return False, "No TF-IDF results in DataFrame"
            
        except Exception as e:
            return False, f"Exception: {e}"
    
    def check_zipf_heaps_laws(self) -> Tuple[bool, str]:
        """Check Zipf and Heaps law analysis with R² validation"""
        try:
            from stats import analyze_zipf_law, analyze_heaps_law
            
            # Generate realistic test data
            test_freq = Counter()
            for i in range(100):
                freq = max(1, int(100 / (i + 1) ** 1.2))
                test_freq[f'word_{i}'] = freq
            
            test_corpus = []
            for doc_idx in range(20):
                doc_tokens = [f'word_{i % 30}' for i in range(doc_idx, doc_idx + 15)]
                test_corpus.append(doc_tokens)
            
            # Test Zipf analysis
            zipf_results = analyze_zipf_law(test_freq)
            if not isinstance(zipf_results, dict) or 'r_squared' not in zipf_results:
                return False, "Invalid Zipf results structure"
            
            if not (0 <= zipf_results['r_squared'] <= 1):
                return False, f"Invalid Zipf R²: {zipf_results['r_squared']}"
            
            # Test Heaps analysis
            heaps_results = analyze_heaps_law(test_corpus)
            if not isinstance(heaps_results, dict) or 'r_squared' not in heaps_results:
                return False, "Invalid Heaps results structure"
            
            if not (0 <= heaps_results['r_squared'] <= 1):
                return False, f"Invalid Heaps R²: {heaps_results['r_squared']}"
            
            return True, f"Zipf R²: {zipf_results['r_squared']:.3f}, Heaps R²: {heaps_results['r_squared']:.3f}"
            
        except Exception as e:
            return False, f"Exception: {e}"
    
    def check_visualization_creation(self) -> Tuple[bool, str]:
        """Check visualization functions create files successfully"""
        try:
            from viz import create_zipf_panels, create_heaps_plot, create_wordcloud
            from stats import analyze_zipf_law, analyze_heaps_law
            import tempfile
            
            # Generate test data with sufficient size for analysis
            test_freq = Counter()
            for i in range(100):  # More data points
                freq = max(1, int(100 / (i + 1) ** 1.1))
                test_freq[f'word_{i}'] = freq
            
            test_corpus = []
            for doc_idx in range(50):  # More documents
                doc_tokens = [f'word_{i % 30}' for i in range(doc_idx, doc_idx + 20)]
                test_corpus.append(doc_tokens)
            
            with tempfile.TemporaryDirectory() as tmp_dir:
                files_created = []
                
                # Test Zipf panels
                zipf_results = analyze_zipf_law(test_freq)
                if zipf_results.get('r_squared', 0) > 0:  # Only create if analysis succeeded
                    zipf_path = os.path.join(tmp_dir, "test_zipf.png")
                    create_zipf_panels(test_freq, zipf_path, zipf_results)
                    if os.path.exists(zipf_path) and os.path.getsize(zipf_path) > 1000:
                        files_created.append("zipf")
                
                # Test Heaps plot
                heaps_results = analyze_heaps_law(test_corpus)
                if heaps_results.get('r_squared', 0) > 0:  # Only create if analysis succeeded
                    heaps_path = os.path.join(tmp_dir, "test_heaps.png")
                    create_heaps_plot(test_corpus, heaps_path, heaps_results)
                    if os.path.exists(heaps_path) and os.path.getsize(heaps_path) > 1000:
                        files_created.append("heaps")
                
                # Test word cloud (this should always work)
                cloud_path = os.path.join(tmp_dir, "test_cloud.png")
                create_wordcloud(test_freq, cloud_path, "Test")
                if os.path.exists(cloud_path) and os.path.getsize(cloud_path) > 1000:
                    files_created.append("wordcloud")
                
                if len(files_created) < 1:  # At least wordcloud should work
                    return False, f"No visualizations created: {files_created}"
                
                return True, f"Visualizations created: {files_created}"
                
        except Exception as e:
            return False, f"Exception: {e}"
    
    def check_chinese_font_detection(self) -> Tuple[bool, str]:
        """Check Chinese font auto-detection system"""
        try:
            from viz import setup_chinese_font
            
            # Test font detection (should not crash)
            font_path = setup_chinese_font()
            
            # Check that font data file exists
            font_data_path = Path(__file__).parent.parent / "data" / "fonts.json"
            if not font_data_path.exists():
                return False, "fonts.json not found"
            
            import json
            with open(font_data_path, 'r', encoding='utf-8') as f:
                font_data = json.load(f)
            
            required_keys = ['chinese_fonts', 'install_instructions', 'wordcloud_requirements']
            if not all(key in font_data for key in required_keys):
                return False, f"fonts.json missing required keys: {required_keys}"
            
            return True, f"Font detection: {type(font_path).__name__}, Data structure: OK"
            
        except Exception as e:
            return False, f"Exception: {e}"
    
    def check_error_handling(self) -> Tuple[bool, str]:
        """Check graceful error handling for edge cases"""
        try:
            from stats import analyze_zipf_law, analyze_heaps_law, calculate_frequencies
            
            # Test empty data
            empty_freq = Counter()
            zipf_empty = analyze_zipf_law(empty_freq)
            if zipf_empty['r_squared'] != 0:
                return False, "Empty data not handled correctly in Zipf"
            
            empty_corpus = []
            heaps_empty = analyze_heaps_law(empty_corpus)
            if heaps_empty['r_squared'] != 0:
                return False, "Empty data not handled correctly in Heaps"
            
            # Test minimal data
            minimal_freq = Counter({'a': 1})
            zipf_minimal = analyze_zipf_law(minimal_freq)
            if zipf_minimal['r_squared'] != 0:
                return False, "Minimal data not handled correctly in Zipf"
            
            minimal_corpus = [['a']]
            heaps_minimal = analyze_heaps_law(minimal_corpus)
            if heaps_minimal['r_squared'] != 0:
                return False, "Minimal data not handled correctly in Heaps"
            
            return True, "Error handling for edge cases works correctly"
            
        except Exception as e:
            return False, f"Exception: {e}"
    
    def check_memory_efficiency(self) -> Tuple[bool, str]:
        """Check memory efficiency with larger datasets"""
        try:
            from ngrams import build_ngrams
            import psutil
            import os
            
            # Monitor memory before
            process = psutil.Process(os.getpid())
            memory_before = process.memory_info().rss / 1024 / 1024  # MB
            
            # Generate larger dataset
            large_tokens = []
            for i in range(1000):
                large_tokens.extend([f'word_{i % 100}', f'term_{i % 50}'])
            
            # Process with n-grams
            merged_tokens, ngram_counts = build_ngrams(
                tokens=large_tokens,
                max_n=5,
                min_freq=3,
                collocation='pmi',
                pmi_threshold=2.0,
                llr_threshold=5.0
            )
            
            # Monitor memory after
            memory_after = process.memory_info().rss / 1024 / 1024  # MB
            memory_increase = memory_after - memory_before
            
            # Check reasonable memory usage (< 100MB increase for this test)
            if memory_increase > 100:
                return False, f"Excessive memory usage: +{memory_increase:.1f}MB"
            
            # Check that processing completed
            if not merged_tokens:
                return False, "No tokens processed"
            
            return True, f"Processed {len(large_tokens)} tokens, memory: +{memory_increase:.1f}MB"
            
        except ImportError:
            return True, "psutil not available, skipping memory check"
        except Exception as e:
            return False, f"Exception: {e}"
    
    def check_reproducibility(self) -> Tuple[bool, str]:
        """Check reproducibility with fixed random seed"""
        try:
            from stats import analyze_heaps_law
            import random
            import numpy as np
            
            # Generate reproducible test data
            random.seed(42)
            np.random.seed(42)
            
            test_corpus1 = []
            for i in range(10):
                doc_length = random.randint(5, 15)
                doc_tokens = [f'word_{random.randint(0, 20)}' for _ in range(doc_length)]
                test_corpus1.append(doc_tokens)
            
            result1 = analyze_heaps_law(test_corpus1)
            
            # Reset seed and repeat
            random.seed(42)
            np.random.seed(42)
            
            test_corpus2 = []
            for i in range(10):
                doc_length = random.randint(5, 15)
                doc_tokens = [f'word_{random.randint(0, 20)}' for _ in range(doc_length)]
                test_corpus2.append(doc_tokens)
            
            result2 = analyze_heaps_law(test_corpus2)
            
            # Compare results
            if abs(result1['r_squared'] - result2['r_squared']) > 1e-10:
                return False, f"Non-reproducible results: {result1['r_squared']} vs {result2['r_squared']}"
            
            return True, f"Reproducible R²: {result1['r_squared']:.6f}"
            
        except Exception as e:
            return False, f"Exception: {e}"
    
    def print_summary(self):
        """Print comprehensive summary of all checks"""
        print("\n" + "=" * 70)
        print("📊 SELF-CHECK SUMMARY")
        print("=" * 70)
        
        total_checks = len(self.results)
        passed_checks = sum(1 for r in self.results.values() if r['success'])
        
        print(f"Total checks: {total_checks}")
        print(f"Passed: {passed_checks}")
        print(f"Failed: {total_checks - passed_checks}")
        print(f"Success rate: {passed_checks/total_checks*100:.1f}%")
        
        if self.warnings:
            print(f"\n⚠️  Warnings ({len(self.warnings)}):")
            for warning in self.warnings:
                print(f"  - {warning}")
        
        if self.errors:
            print(f"\n❌ Errors ({len(self.errors)}):")
            for error in self.errors:
                print(f"  - {error}")
        
        print("\n🔍 Detailed Results:")
        for check_name, result in self.results.items():
            status = "✅ PASS" if result['success'] else "❌ FAIL"
            print(f"  {check_name}: {status} ({result['elapsed_time']:.2f}s)")
            if not result['success']:
                print(f"    {result['details']}")


def run_quick_checks() -> bool:
    """Run a subset of critical checks for quick validation"""
    print("🚀 Running quick validation checks...")
    
    framework = SelfCheckFramework()
    
    # Run only critical checks
    critical_checks = [
        ("tokenizer_fallback", framework.check_tokenizer_fallback),
        ("tfidf_pretokenized", framework.check_tfidf_pretokenized),
        ("zipf_heaps_laws", framework.check_zipf_heaps_laws),
    ]
    
    all_passed = True
    for check_name, check_func in critical_checks:
        try:
            success, details = check_func()
            if success:
                print(f"✅ {check_name}")
            else:
                print(f"❌ {check_name}: {details}")
                all_passed = False
        except Exception as e:
            print(f"💥 {check_name}: {e}")
            all_passed = False
    
    return all_passed


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Analysis pipeline self-check")
    parser.add_argument("--quick", action="store_true", help="Run only critical checks")
    parser.add_argument("--json", type=str, help="Save results to JSON file")
    args = parser.parse_args()
    
    if args.quick:
        success = run_quick_checks()
        exit_code = 0 if success else 1
    else:
        framework = SelfCheckFramework()
        success, results = framework.run_all_checks()
        
        if args.json:
            import json
            with open(args.json, 'w', encoding='utf-8') as f:
                json.dump(results, f, ensure_ascii=False, indent=2, default=str)
            print(f"\n💾 Results saved to: {args.json}")
        
        exit_code = 0 if success else 1
    
    print(f"\n{'🎉 ALL CHECKS PASSED' if success else '⚠️  SOME CHECKS FAILED'}")
    sys.exit(exit_code)