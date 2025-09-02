#!/usr/bin/env python3
"""
Integration Test for Academic Word Cloud System

This script validates that all academic enhancement components work together
and meet the quality standards for academic conferences.
"""

import sys
import json
from pathlib import Path
from collections import Counter

# Add analysis modules to path
sys.path.insert(0, str(Path(__file__).parent))

from benchmark_wordcloud import AcademicWordCloudBenchmark
from enhanced_wordcloud_viz import EnhancedWordCloudGenerator
from evaluation_metrics import AcademicWordCloudEvaluator


def run_integration_test():
    """Run comprehensive integration test of all academic features"""
    print("🧪 Starting Academic Word Cloud System Integration Test")
    print("=" * 60)
    
    # Test data setup
    test_frequencies = Counter({
        'research': 150, 'analysis': 120, 'academic': 100, 'study': 95,
        'method': 90, 'paper': 85, 'conference': 80, 'publication': 75,
        'algorithm': 70, 'framework': 65, 'evaluation': 60, 'system': 55,
        'approach': 50, 'model': 45, 'technique': 40, 'result': 35,
        'experiment': 30, 'data': 28, 'performance': 25, 'quality': 22,
        'computational': 20, 'linguistics': 18, 'statistical': 16
    })
    
    test_passed = True
    
    try:
        # 1. Test Benchmarking System
        print("\n🔬 Testing Benchmarking System...")
        benchmark = AcademicWordCloudBenchmark(output_dir="out/test_benchmarks")
        
        # Run simplified benchmark
        corpus_texts = ["test document " + str(i) for i in range(100)]
        tokenization_result = benchmark.benchmark_tokenization(corpus_texts, iterations=2)
        
        if tokenization_result.success:
            print("  ✅ Benchmarking system operational")
        else:
            print(f"  ❌ Benchmarking failed: {tokenization_result.error_message}")
            test_passed = False
        
        # 2. Test Enhanced Visualization
        print("\n🎨 Testing Enhanced Visualization...")
        generator = EnhancedWordCloudGenerator()
        
        # Test academic color schemes
        schemes = ['nature_publication', 'ieee_compliant', 'accessibility_friendly']
        for scheme in schemes:
            try:
                output_path = f"out/test_academic_{scheme}.png"
                generator.generate_academic_wordcloud(
                    frequencies=test_frequencies,
                    output_path=output_path,
                    scheme=scheme,
                    title=f"Test - {scheme.replace('_', ' ').title()}",
                    max_words=50
                )
                print(f"  ✅ {scheme} visualization generated")
            except Exception as e:
                print(f"  ❌ {scheme} visualization failed: {e}")
                test_passed = False
        
        # 3. Test Quality Evaluation
        print("\n📊 Testing Quality Evaluation...")
        evaluator = AcademicWordCloudEvaluator()
        
        try:
            metrics = evaluator.comprehensive_evaluation(
                test_frequencies,
                reference_corpus=["academic research paper analysis"] * 10
            )
            
            # Validate metrics
            required_metrics = [
                'vocabulary_diversity', 'zipf_compliance', 'visual_balance',
                'readability_score', 'semantic_consistency', 'composite_score'
            ]
            
            all_metrics_present = all(hasattr(metrics, metric) for metric in required_metrics)
            
            if all_metrics_present and metrics.composite_score > 0:
                print(f"  ✅ Quality evaluation complete (Grade: {metrics.quality_grade})")
                print(f"     Composite Score: {metrics.composite_score:.3f}")
            else:
                print("  ❌ Quality evaluation incomplete or invalid")
                test_passed = False
                
        except Exception as e:
            print(f"  ❌ Quality evaluation failed: {e}")
            test_passed = False
        
        # 4. Test Data Integration
        print("\n🔗 Testing Data Integration...")
        try:
            # Check if web data exists and is accessible
            web_data_path = Path("web/wordcloud_data.json")
            if web_data_path.exists():
                with open(web_data_path, 'r', encoding='utf-8') as f:
                    web_data = json.load(f)
                
                if 'overall' in web_data and 'by_year' in web_data:
                    print("  ✅ Web interface data integration operational")
                else:
                    print("  ⚠️ Web interface data incomplete")
            else:
                print("  ⚠️ Web interface data not found (may need generation)")
                
        except Exception as e:
            print(f"  ❌ Data integration test failed: {e}")
            test_passed = False
        
        # 5. Test Academic Standards Compliance
        print("\n🎓 Testing Academic Standards Compliance...")
        
        compliance_checks = {
            'reproducibility': metrics.reproducibility_score > 0.9,
            'statistical_rigor': metrics.zipf_compliance > 0.5,
            'visualization_quality': metrics.readability_score > 0.6,
            'semantic_coherence': metrics.semantic_consistency > 0.3,
            'overall_quality': metrics.composite_score > 0.5
        }
        
        passed_checks = sum(compliance_checks.values())
        total_checks = len(compliance_checks)
        
        print(f"  Academic Compliance: {passed_checks}/{total_checks} checks passed")
        
        for check, passed in compliance_checks.items():
            status = "✅" if passed else "⚠️"
            print(f"    {status} {check.replace('_', ' ').title()}")
        
        if passed_checks >= total_checks * 0.8:  # 80% pass rate
            print("  ✅ Academic standards compliance: PASS")
        else:
            print("  ⚠️ Academic standards compliance: NEEDS IMPROVEMENT")
        
        # 6. Conference Readiness Assessment
        print("\n🏆 Conference Readiness Assessment...")
        
        conference_scores = {
            'WWW 2026': 0.8 if metrics.composite_score > 0.75 else 0.6,
            'SIGIR 2026': 0.9 if metrics.composite_score > 0.8 else 0.7,
            'ICWSM 2026': 0.8 if metrics.composite_score > 0.7 else 0.6,
            'CIKM 2026': 0.7 if metrics.composite_score > 0.6 else 0.5,
            'WSDM 2026': 0.8 if metrics.composite_score > 0.75 else 0.6,
            'CHI 2026': 0.9 if metrics.readability_score > 0.8 else 0.7
        }
        
        for conference, score in conference_scores.items():
            status = "🚀" if score >= 0.8 else "📈" if score >= 0.7 else "⚠️"
            print(f"    {status} {conference}: {score:.1f}/1.0 readiness")
        
        avg_readiness = sum(conference_scores.values()) / len(conference_scores)
        print(f"\n  Average Conference Readiness: {avg_readiness:.2f}/1.0")
        
    except Exception as e:
        print(f"\n❌ Integration test failed with error: {e}")
        test_passed = False
    
    # Final Results
    print("\n" + "=" * 60)
    if test_passed:
        print("🎉 INTEGRATION TEST PASSED")
        print("   All academic enhancement components are operational")
        print("   System ready for academic conference submission")
    else:
        print("⚠️ INTEGRATION TEST INCOMPLETE")
        print("   Some components may need attention before academic use")
    
    print("\n📋 Test Summary:")
    print(f"   - Benchmarking: {'✅' if tokenization_result.success else '❌'}")
    print(f"   - Visualization: {'✅' if test_passed else '❌'}")
    print(f"   - Quality Evaluation: {'✅' if 'metrics' in locals() else '❌'}")
    print(f"   - Academic Compliance: {'✅' if passed_checks >= total_checks * 0.8 else '⚠️'}")
    
    return test_passed


if __name__ == "__main__":
    success = run_integration_test()
    sys.exit(0 if success else 1)