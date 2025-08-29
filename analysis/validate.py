#!/usr/bin/env python3
"""
Quick validation script to test the implementation structure
"""

import sys
from pathlib import Path

# Add pipeline to path
sys.path.insert(0, str(Path(__file__).parent / "pipeline"))

def test_imports():
    """Test if all modules can be imported"""
    print("🧪 Testing module imports...")
    
    try:
        # Test corpus_io
        from corpus_io import Article, load_corpus, read_text, get_corpus_stats
        print("✅ corpus_io module imported successfully")
        
        # Test basic tokenizer without jieba
        from tokenize import ChineseNormalizer, TokenizerProtocol
        print("✅ tokenize module basic classes imported successfully")
        
        # Test ngrams
        from ngrams import build_ngrams, get_ngram_stats
        print("✅ ngrams module imported successfully")
        
        # Test stats (basic functions)
        from stats import calculate_frequencies, analyze_zipf_law, analyze_heaps_law
        print("✅ stats module imported successfully")
        
        # Test report
        from report import write_report
        print("✅ report module imported successfully")
        
        print("🎉 All core modules imported successfully!")
        return True
        
    except ImportError as e:
        print(f"❌ Import error: {e}")
        return False


def test_data_files():
    """Test if data files exist"""
    print("\n📁 Testing data files...")
    
    data_files = [
        "data/stopwords.zh.txt",
        "data/stopwords.en.txt", 
        "data/allow_singletons.zh.txt"
    ]
    
    all_exist = True
    for file_path in data_files:
        if Path(file_path).exists():
            print(f"✅ {file_path} exists")
        else:
            print(f"❌ {file_path} missing")
            all_exist = False
    
    return all_exist


def test_structure():
    """Test directory structure"""
    print("\n🏗️  Testing directory structure...")
    
    expected_structure = [
        "pipeline/corpus_io.py",
        "pipeline/tokenize.py",
        "pipeline/ngrams.py",
        "pipeline/stats.py",
        "pipeline/viz.py",
        "pipeline/report.py",
        "data/",
        "out/",
        "main.py"
    ]
    
    all_exist = True
    for path in expected_structure:
        if Path(path).exists():
            print(f"✅ {path} exists")
        else:
            print(f"❌ {path} missing")
            all_exist = False
    
    return all_exist


def test_main_config():
    """Test main.py configuration structure"""
    print("\n⚙️  Testing main.py configuration...")
    
    try:
        with open("main.py", 'r', encoding='utf-8') as f:
            content = f.read()
        
        required_configs = [
            "RUN_ANALYSIS",
            "RUN_VISUALIZATION", 
            "MAX_N",
            "MIN_FREQ",
            "COLLOCATION",
            "TOKENIZER_TYPE"
        ]
        
        all_configs = True
        for config in required_configs:
            if config in content:
                print(f"✅ {config} configuration found")
            else:
                print(f"❌ {config} configuration missing")
                all_configs = False
        
        return all_configs
        
    except Exception as e:
        print(f"❌ Error reading main.py: {e}")
        return False


def main():
    """Run all validation tests"""
    print("=" * 60)
    print("🔬 LINGUISTIC ANALYSIS IMPLEMENTATION VALIDATION")
    print("=" * 60)
    
    tests = [
        ("Module Imports", test_imports),
        ("Data Files", test_data_files),
        ("Directory Structure", test_structure),
        ("Main Configuration", test_main_config)
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
    print("📊 VALIDATION SUMMARY")
    print("=" * 60)
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for test_name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{status} {test_name}")
    
    print(f"\n🎯 Overall: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All validations passed! Implementation structure is correct.")
        print("📝 Next step: Install dependencies and run analysis")
    else:
        print("⚠️  Some validations failed. Check the issues above.")
    
    return passed == total


if __name__ == "__main__":
    main()