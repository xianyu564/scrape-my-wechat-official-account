#!/usr/bin/env python3
"""
Quick test for performance optimizations
"""

import sys
import os
from pathlib import Path

# Add pipeline to path
sys.path.insert(0, str(Path(__file__).parent / "pipeline"))

from performance import profiler, optimize_memory, benchmark_function
from stats import calculate_lexical_metrics, _bootstrap_heaps_confidence
from advanced_analysis import calculate_advanced_metrics
import numpy as np
from collections import Counter

def test_performance_optimizations():
    """Test the performance optimizations"""
    print("🧪 Testing Performance Optimizations")
    print("=" * 50)
    
    # Generate test data
    print("📊 Generating test data...")
    test_corpus = [
        ["这是", "一个", "测试", "文档"],
        ["测试", "文档", "包含", "中文", "词汇"],
        ["性能", "优化", "测试", "正在", "进行"],
        ["bootstrap", "confidence", "intervals", "testing"],
        ["advanced", "metrics", "calculation", "verification"]
    ]
    
    # Test 1: Lexical metrics optimization
    print("\n🔧 Testing optimized lexical metrics...")
    
    @profiler.profile(include_memory=True)
    def test_lexical_metrics():
        return calculate_lexical_metrics(test_corpus)
    
    metrics = test_lexical_metrics()
    print(f"✅ Lexical metrics calculated: TTR={metrics.get('ttr', 0):.3f}")
    
    # Test 2: Bootstrap confidence optimization
    print("\n🔢 Testing bootstrap confidence intervals...")
    
    log_n = np.log(np.array([10, 20, 30, 40, 50]))
    log_V = np.log(np.array([8, 15, 20, 24, 27]))
    
    @profiler.profile(include_memory=True)
    def test_bootstrap():
        return _bootstrap_heaps_confidence(log_n, log_V, n_bootstrap=20)
    
    conf_lower, conf_upper = test_bootstrap()
    print(f"✅ Bootstrap CI: [{conf_lower:.3f}, {conf_upper:.3f}]")
    
    # Test 3: Advanced metrics
    print("\n📈 Testing advanced metrics...")
    
    @profiler.profile(include_memory=True)
    def test_advanced_metrics():
        return calculate_advanced_metrics(test_corpus)
    
    advanced = test_advanced_metrics()
    print(f"✅ Advanced metrics calculated: {len(advanced)} metrics")
    
    # Test 4: Memory optimization
    print("\n🧹 Testing memory optimization...")
    optimize_memory()
    
    # Performance summary
    print("\n📊 Performance Summary:")
    profiler.print_summary()
    
    return True

if __name__ == "__main__":
    try:
        success = test_performance_optimizations()
        if success:
            print("\n🎉 All performance optimization tests passed!")
        else:
            print("\n❌ Some tests failed")
            sys.exit(1)
    except Exception as e:
        print(f"\n💥 Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)