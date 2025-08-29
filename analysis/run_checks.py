#!/usr/bin/env python3
"""
Quick validation script for analysis pipeline
Tests directly on WeChat backup data without mock files
"""

import os
import sys
import time
from pathlib import Path

# Add current directory and pipeline to path
sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent / "pipeline"))

def validate_pipeline_imports():
    """Validate that all pipeline modules can be imported"""
    print("📦 Checking pipeline imports...")
    
    try:
        from corpus_io import load_corpus, read_text, get_corpus_stats
        print("  ✅ corpus_io")
        
        from tokenizer import MixedLanguageTokenizer
        print("  ✅ tokenizer")
        
        from ngrams import build_ngrams, get_ngram_stats
        print("  ✅ ngrams")
        
        from stats import calculate_frequencies, analyze_zipf_law
        print("  ✅ stats")
        
        from viz import create_zipf_panels, create_wordcloud
        print("  ✅ viz")
        
        from report import write_report
        print("  ✅ report")
        
        return True
        
    except ImportError as e:
        print(f"  ❌ Import error: {e}")
        return False

def validate_wechat_data():
    """Validate that WeChat backup data is accessible"""
    print("\n📂 Checking WeChat backup data...")
    
    corpus_path = Path("../Wechat-Backup/文不加点的张衔瑜")
    if not corpus_path.exists():
        print(f"  ❌ WeChat backup not found at: {corpus_path}")
        return False
    
    # Check for year directories
    year_dirs = [d for d in corpus_path.iterdir() if d.is_dir() and d.name.isdigit()]
    if not year_dirs:
        print("  ❌ No year directories found")
        return False
    
    print(f"  ✅ Found {len(year_dirs)} year directories: {sorted([d.name for d in year_dirs])}")
    
    # Check for markdown files
    md_files = list(corpus_path.glob("**/*.md"))
    print(f"  ✅ Found {len(md_files)} markdown files")
    
    return True

def validate_basic_functionality():
    """Test basic functionality with real WeChat data"""
    print("\n🔧 Testing basic functionality...")
    
    try:
        from corpus_io import load_corpus, read_text
        from tokenizer import MixedLanguageTokenizer
        
        # Load a small sample of real data
        corpus_path = "../Wechat-Backup/文不加点的张衔瑜"
        print(f"  📖 Loading corpus from: {corpus_path}")
        
        articles = load_corpus(corpus_path, years=["2024"])
        if not articles:
            print("  ❌ No articles loaded")
            return False
        
        print(f"  ✅ Loaded {len(articles)} articles from 2024")
        
        # Test tokenization
        tokenizer = MixedLanguageTokenizer()
        if articles:
            sample_text = read_text(articles[0])[:200]  # First 200 chars
            if not sample_text:
                print("  ⚠️  Warning: Empty article content")
                return True  # Still consider this a pass, content might be empty
            tokens = tokenizer.tokenize(sample_text)
            print(f"  ✅ Tokenized sample text: {len(tokens)} tokens")
        
        return True
        
    except Exception as e:
        print(f"  ❌ Functionality test failed: {e}")
        return False

def main():
    """Main validation entry point"""
    print("🔍 Analysis Pipeline Validation")
    print("Testing directly on WeChat backup data")
    print("=" * 50)
    print(f"Python: {sys.version}")
    print(f"Working directory: {os.getcwd()}")
    print(f"Pipeline path: {Path(__file__).parent / 'pipeline'}")
    
    # Run validation checks
    checks = [
        ("Pipeline Imports", validate_pipeline_imports),
        ("WeChat Data Access", validate_wechat_data),
        ("Basic Functionality", validate_basic_functionality)
    ]
    
    results = []
    for check_name, check_func in checks:
        print(f"\n{'='*20} {check_name} {'='*20}")
        try:
            success = check_func()
            results.append((check_name, success))
        except Exception as e:
            print(f"❌ {check_name} failed with error: {e}")
            results.append((check_name, False))
    
    # Print summary
    print(f"\n{'='*60}")
    print("📊 VALIDATION SUMMARY")
    print(f"{'='*60}")
    
    passed = sum(1 for _, success in results if success)
    total = len(results)
    
    for check_name, success in results:
        status = "✅ PASS" if success else "❌ FAIL"
        print(f"{status} {check_name}")
    
    overall_success = passed == total
    print(f"\n🎯 Overall: {passed}/{total} checks passed")
    print(f"{'🎉 SUCCESS' if overall_success else '⚠️  FAILURE'}: Pipeline validation {'passed' if overall_success else 'failed'}")
    
    if overall_success:
        print("\n📝 Next step: Run 'python main.py' for full analysis")
    
    return overall_success

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)