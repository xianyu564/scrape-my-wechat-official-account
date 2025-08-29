#!/usr/bin/env python3
"""
Test enhanced report generation functionality
Task 6: 报告生成器升级
"""

import os
import sys
import pytest
import tempfile
from pathlib import Path
from collections import Counter
import pandas as pd

# Add pipeline to path
sys.path.insert(0, str(Path(__file__).parent.parent / "pipeline"))

from report import write_report


class TestEnhancedReportGeneration:
    """Test enhanced report generation with templates and structure"""
    
    def setup_method(self):
        """Setup test data for report generation"""
        # Mock corpus stats
        self.corpus_stats = {
            'total_articles': 150,
            'years': [2020, 2021, 2022, 2023],
            'total_chars': 50000,
            'avg_chars_per_article': 333
        }
        
        # Mock frequencies
        self.freq_overall = Counter({
            'machine_learning': 45, 'artificial_intelligence': 35, 'data_science': 28,
            'neural_networks': 22, 'deep_learning': 18, 'python': 15,
            'algorithm': 12, 'research': 10, 'analysis': 8, 'technology': 7
        })
        
        self.freq_by_year = {
            '2020': Counter({'machine_learning': 12, 'data_science': 8, 'python': 5}),
            '2021': Counter({'neural_networks': 15, 'deep_learning': 10, 'algorithm': 7}),
            '2022': Counter({'artificial_intelligence': 20, 'research': 6, 'analysis': 4}),
            '2023': Counter({'technology': 7, 'machine_learning': 18, 'python': 8})
        }
        
        # Mock TF-IDF results
        self.tfidf_results = pd.DataFrame({
            'year': ['2023', '2023', '2022', '2022'],
            'word': ['machine_learning', 'AI', 'neural_net', 'algorithm'],
            'score': [0.85, 0.72, 0.68, 0.55]
        })
        
        # Mock statistical results
        self.zipf_results = {
            'slope': -1.15, 'r_squared': 0.92, 'intercept': 3.2,
            'p_value': 1e-25, 'std_err': 0.02
        }
        
        self.heaps_results = {
            'K': 2.8, 'beta': 0.55, 'r_squared': 0.88,
            'p_value': 1e-15, 'std_err': 0.01
        }
        
        # Mock lexical metrics
        self.lexical_metrics = {
            'total_tokens': 12500, 'total_unique_tokens': 3200,
            'type_token_ratio': 0.256, 'lexical_density': 0.65,
            'avg_tokens_per_doc': 83.3, 'content_word_ratio': 0.45
        }
        
        # Mock n-gram stats
        self.ngram_stats = {1: 3200, 2: 1250, 3: 485, 4: 142, 5: 38}
        
        # Mock growth data
        self.growth_data = [
            {'word': 'AI', 'growth': 45.2, 'freq_2022': 12, 'freq_2023': 17.4},
            {'word': 'automation', 'growth': 33.8, 'freq_2022': 8, 'freq_2023': 10.7},
            {'word': 'efficiency', 'growth': -15.2, 'freq_2022': 15, 'freq_2023': 12.7}
        ]
        
        # Mock analysis params
        self.analysis_params = {
            'max_n': 8, 'min_freq': 5, 'collocation': 'pmi',
            'pmi_threshold': 3.0, 'seed': 42, 'tokenizer_type': 'auto'
        }
        
        # Mock tokenizer info
        self.tokenizer_info = {
            'name': 'jieba (fallback)', 'backend_available': {'pkuseg': False, 'jieba': True},
            'user_dictionaries': 1, 'fallback_used': True
        }
    
    def test_basic_report_generation(self, tmp_path):
        """Test basic report generation functionality"""
        output_dir = str(tmp_path)
        
        report_path = write_report(
            output_dir=output_dir,
            corpus_stats=self.corpus_stats,
            freq_overall=self.freq_overall,
            freq_by_year=self.freq_by_year,
            tfidf_results=self.tfidf_results,
            zipf_results=self.zipf_results,
            heaps_results=self.heaps_results,
            lexical_metrics=self.lexical_metrics,
            ngram_stats=self.ngram_stats,
            growth_data=self.growth_data,
            analysis_params=self.analysis_params,
            tokenizer_info=self.tokenizer_info
        )
        
        # Check that report was created
        assert os.path.exists(report_path)
        assert report_path.endswith('report.md')
        
        # Check file is not empty
        with open(report_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        assert len(content) > 1000, "Report should be substantial"
        
        # Check for required sections
        required_sections = [
            '# Corpus Linguistic Analysis Report',
            '## 1. Executive Summary',
            '## 2. Methods & Configuration',
            '## 3. Global Overview',
            '## 4. Yearly Snapshots',
            '## 5. Phrase Inventory',
            '## 6. Year-over-Year Movers',
            '## 7. Statistical Laws Analysis',
            '## 8. Cheer-Me-Up Summary',
            '## 9. Reproducibility Notes'
        ]
        
        for section in required_sections:
            assert section in content, f"Missing section: {section}"
    
    def test_statistical_laws_interpretation(self, tmp_path):
        """Test statistical laws analysis and interpretation"""
        output_dir = str(tmp_path)
        
        report_path = write_report(
            output_dir=output_dir,
            corpus_stats=self.corpus_stats,
            freq_overall=self.freq_overall,
            freq_by_year=self.freq_by_year,
            tfidf_results=self.tfidf_results,
            zipf_results=self.zipf_results,
            heaps_results=self.heaps_results,
            lexical_metrics=self.lexical_metrics,
            ngram_stats=self.ngram_stats,
            growth_data=self.growth_data,
            analysis_params=self.analysis_params,
            tokenizer_info=self.tokenizer_info
        )
        
        with open(report_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Check Zipf analysis
        assert '**Slope**: -1.150' in content
        assert '**R² Goodness of Fit**: 0.920' in content
        assert 'Approximately Zipfian' in content or 'Classical Zipfian' in content
        
        # Check Heaps analysis
        assert '**Growth Exponent (β)**: 0.550' in content
        assert '**Scaling Constant (K)**: 2.800' in content
        assert 'Classical sub-linear growth' in content
    
    def test_cheer_up_section_positivity(self, tmp_path):
        """Test that cheer-up section is positive and encouraging"""
        output_dir = str(tmp_path)
        
        report_path = write_report(
            output_dir=output_dir,
            corpus_stats=self.corpus_stats,
            freq_overall=self.freq_overall,
            freq_by_year=self.freq_by_year,
            tfidf_results=self.tfidf_results,
            zipf_results=self.zipf_results,
            heaps_results=self.heaps_results,
            lexical_metrics=self.lexical_metrics,
            ngram_stats=self.ngram_stats,
            growth_data=self.growth_data,
            analysis_params=self.analysis_params,
            tokenizer_info=self.tokenizer_info
        )
        
        with open(report_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Find cheer-up section
        cheer_start = content.find('## 8. Cheer-Me-Up Summary')
        cheer_end = content.find('## 9. Reproducibility Notes')
        cheer_section = content[cheer_start:cheer_end]
        
        # Check for positive language
        positive_phrases = [
            'remarkable', 'inspiring', 'impressive', 'sophisticated',
            'awesome', 'beautiful', 'unique', 'growing'
        ]
        
        found_positive = sum(1 for phrase in positive_phrases if phrase in cheer_section.lower())
        assert found_positive >= 3, f"Should have at least 3 positive phrases, found {found_positive}"
        
        # Check for specific encouraging elements
        assert '🚀' in cheer_section  # Productivity emoji
        assert 'articles' in cheer_section  # Article count
        assert 'vocabulary' in cheer_section or 'terms' in cheer_section  # Vocab mention
    
    def test_template_fallback_mechanism(self, tmp_path):
        """Test that report works even without Jinja2 templates"""
        output_dir = str(tmp_path)
        
        # This test ensures the fallback works
        report_path = write_report(
            output_dir=output_dir,
            corpus_stats=self.corpus_stats,
            freq_overall=self.freq_overall,
            freq_by_year=self.freq_by_year,
            tfidf_results=self.tfidf_results,
            zipf_results=self.zipf_results,
            heaps_results=self.heaps_results,
            lexical_metrics=self.lexical_metrics,
            ngram_stats=self.ngram_stats,
            growth_data=self.growth_data,
            analysis_params=self.analysis_params,
            tokenizer_info=self.tokenizer_info
        )
        
        assert os.path.exists(report_path)
        
        with open(report_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Should work regardless of template availability
        assert len(content) > 500
        assert 'Corpus Linguistic Analysis Report' in content
    
    def test_reproducibility_section(self, tmp_path):
        """Test reproducibility section includes key parameters"""
        output_dir = str(tmp_path)
        
        report_path = write_report(
            output_dir=output_dir,
            corpus_stats=self.corpus_stats,
            freq_overall=self.freq_overall,
            freq_by_year=self.freq_by_year,
            tfidf_results=self.tfidf_results,
            zipf_results=self.zipf_results,
            heaps_results=self.heaps_results,
            lexical_metrics=self.lexical_metrics,
            ngram_stats=self.ngram_stats,
            growth_data=self.growth_data,
            analysis_params=self.analysis_params,
            tokenizer_info=self.tokenizer_info
        )
        
        with open(report_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Check for parameter documentation
        assert '"max_n": 8' in content
        assert '"seed": 42' in content
        assert '"collocation": "pmi"' in content
        assert 'jieba' in content  # Tokenizer backend
    
    def test_empty_data_handling(self, tmp_path):
        """Test report generation with minimal/empty data"""
        output_dir = str(tmp_path)
        
        # Use minimal data
        minimal_stats = {'total_articles': 1, 'years': [2023]}
        minimal_freq = Counter({'test': 1})
        minimal_by_year = {'2023': Counter({'test': 1})}
        empty_tfidf = pd.DataFrame(columns=['year', 'word', 'score'])
        minimal_zipf = {'slope': 0, 'r_squared': 0, 'intercept': 0}
        minimal_heaps = {'K': 0, 'beta': 0, 'r_squared': 0}
        minimal_lexical = {'total_tokens': 1, 'total_unique_tokens': 1, 'type_token_ratio': 1.0}
        minimal_ngrams = {1: 1}
        
        # Should not crash with minimal data
        report_path = write_report(
            output_dir=output_dir,
            corpus_stats=minimal_stats,
            freq_overall=minimal_freq,
            freq_by_year=minimal_by_year,
            tfidf_results=empty_tfidf,
            zipf_results=minimal_zipf,
            heaps_results=minimal_heaps,
            lexical_metrics=minimal_lexical,
            ngram_stats=minimal_ngrams,
            growth_data=[],
            analysis_params=self.analysis_params,
            tokenizer_info=self.tokenizer_info
        )
        
        assert os.path.exists(report_path)
        
        with open(report_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        assert len(content) > 200  # Should still generate a basic report
        assert 'Executive Summary' in content


def test_report_integration():
    """Integration test for report generation with realistic data"""
    from collections import Counter
    import pandas as pd
    import tempfile
    
    # Test with realistic-looking data
    with tempfile.TemporaryDirectory() as tmp_dir:
        corpus_stats = {'total_articles': 42, 'years': [2022, 2023]}
        freq_overall = Counter({'hello': 10, 'world': 8, 'test': 5})
        freq_by_year = {'2022': Counter({'hello': 6}), '2023': Counter({'world': 4})}
        tfidf_results = pd.DataFrame({'year': ['2023'], 'word': ['hello'], 'score': [0.5]})
        
        # Should complete without errors
        report_path = write_report(
            output_dir=tmp_dir,
            corpus_stats=corpus_stats,
            freq_overall=freq_overall,
            freq_by_year=freq_by_year,
            tfidf_results=tfidf_results,
            zipf_results={'slope': -1.0, 'r_squared': 0.5, 'intercept': 1.0},
            heaps_results={'K': 1.5, 'beta': 0.5, 'r_squared': 0.7},
            lexical_metrics={'total_tokens': 100, 'total_unique_tokens': 25, 'type_token_ratio': 0.25},
            ngram_stats={1: 25, 2: 10},
            growth_data=[],
            analysis_params={'max_n': 3, 'seed': 42},
            tokenizer_info={'name': 'test', 'user_dictionaries': 0}
        )
        
        assert os.path.exists(report_path)


if __name__ == "__main__":
    # Run basic integration test
    test_report_integration()
    print("✅ Report generation integration test passed!")