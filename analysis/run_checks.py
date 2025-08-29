#!/usr/bin/env python3
"""
Quick validation script for analysis pipeline
Convenient wrapper for running checks without full test environment
"""

import os
import sys
import time
from pathlib import Path

# Add current directory to path
sys.path.insert(0, str(Path(__file__).parent))

def main():
    """Main validation entry point"""
    print("🔍 Analysis Pipeline Validation")
    print("=" * 50)
    print(f"Python: {sys.version}")
    print(f"Working directory: {os.getcwd()}")
    print(f"Pipeline path: {Path(__file__).parent / 'pipeline'}")
    
    try:
        # Import and run self-checks
        from tests.test_selfcheck import SelfCheckFramework, run_quick_checks
        
        # Check command line arguments
        quick_mode = "--quick" in sys.argv
        verbose = "--verbose" in sys.argv or "-v" in sys.argv
        
        if quick_mode:
            print("\n⚡ Running quick checks...")
            success = run_quick_checks()
        else:
            print("\n🧪 Running comprehensive checks...")
            framework = SelfCheckFramework()
            success, results = framework.run_all_checks()
            
            if verbose:
                print("\n📋 Detailed breakdown:")
                for check_name, result in results.items():
                    status = "✅" if result['success'] else "❌"
                    time_str = f"({result['elapsed_time']:.2f}s)"
                    print(f"  {status} {check_name} {time_str}")
                    if not result['success'] and verbose:
                        print(f"      {result['details']}")
        
        # Print final result
        print(f"\n{'🎉 SUCCESS' if success else '⚠️  FAILURE'}: Pipeline validation {'passed' if success else 'failed'}")
        
        return 0 if success else 1
        
    except ImportError as e:
        print(f"❌ Import error: {e}")
        print("💡 Make sure you're running from the analysis/ directory")
        return 1
        
    except Exception as e:
        print(f"💥 Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        return 1


def check_dependencies():
    """Check if required dependencies are available"""
    print("📦 Checking dependencies...")
    
    required_packages = [
        'numpy', 'pandas', 'matplotlib', 'seaborn', 'sklearn', 
        'jieba', 'wordcloud', 'scipy', 'PIL'
    ]
    
    missing = []
    for package in required_packages:
        try:
            if package == 'PIL':
                import PIL
            elif package == 'sklearn':
                import sklearn
            else:
                __import__(package)
            print(f"  ✅ {package}")
        except ImportError:
            print(f"  ❌ {package}")
            missing.append(package)
    
    if missing:
        print(f"\n⚠️  Missing packages: {missing}")
        print("📋 Install with: pip install -r requirements.txt")
        return False
    
    print("✅ All dependencies available")
    return True


def test_basic_functionality():
    """Test basic functionality without full framework"""
    print("\n🧪 Testing basic functionality...")
    
    try:
        # Test imports
        sys.path.insert(0, str(Path(__file__).parent / "pipeline"))
        
        from tokenizer import MixedLanguageTokenizer
        from stats import analyze_zipf_law, analyze_heaps_law
        from viz import create_wordcloud
        
        print("  ✅ Core imports successful")
        
        # Test tokenizer
        tokenizer = MixedLanguageTokenizer(tokenizer_type="auto")
        tokens = tokenizer.tokenize("Hello world 你好世界")
        if tokens:
            print(f"  ✅ Tokenization works: {len(tokens)} tokens")
        else:
            print("  ❌ Tokenization failed")
            return False
        
        # Test basic analysis
        from collections import Counter
        test_freq = Counter({'hello': 10, 'world': 5, 'test': 3})
        zipf_result = analyze_zipf_law(test_freq)
        if 'r_squared' in zipf_result:
            print(f"  ✅ Statistical analysis works")
        else:
            print("  ❌ Statistical analysis failed")
            return False
        
        print("✅ Basic functionality test passed")
        return True
        
    except Exception as e:
        print(f"  ❌ Basic test failed: {e}")
        return False


def print_help():
    """Print help information"""
    help_text = """
Analysis Pipeline Validation Script

Usage:
    python run_checks.py [options]

Options:
    --quick     Run only critical checks (faster)
    --verbose   Show detailed output
    --deps      Check dependencies only
    --basic     Run basic functionality test only
    --help      Show this help

Examples:
    python run_checks.py                    # Full validation
    python run_checks.py --quick            # Quick validation
    python run_checks.py --verbose          # Detailed output
    python run_checks.py --deps --basic     # Check dependencies and basic functionality
"""
    print(help_text)


if __name__ == "__main__":
    # Handle help
    if "--help" in sys.argv or "-h" in sys.argv:
        print_help()
        sys.exit(0)
    
    start_time = time.time()
    
    # Handle specific options
    if "--deps" in sys.argv:
        deps_ok = check_dependencies()
        if not deps_ok:
            sys.exit(1)
        if "--basic" not in sys.argv:
            sys.exit(0)
    
    if "--basic" in sys.argv:
        basic_ok = test_basic_functionality()
        sys.exit(0 if basic_ok else 1)
    
    # Run main validation
    exit_code = main()
    
    elapsed = time.time() - start_time
    print(f"\n⏱️  Total time: {elapsed:.1f}s")
    
    sys.exit(exit_code)