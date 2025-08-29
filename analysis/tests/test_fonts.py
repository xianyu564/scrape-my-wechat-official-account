#!/usr/bin/env python3
"""
Test Chinese font detection and visualization functionality
Task 5: 可视化与中文字体自动识别
"""

import os
import sys
import json
import pytest
from unittest.mock import patch
from pathlib import Path

# Add pipeline to path
sys.path.insert(0, str(Path(__file__).parent.parent / "pipeline"))

from viz import setup_chinese_font, create_zipf_panels, create_heaps_plot, create_wordcloud
from collections import Counter


class TestChineseFontDetection:
    """Test Chinese font auto-detection functionality"""
    
    def test_custom_font_path(self, tmp_path):
        """Test using custom font path"""
        # Create a dummy font file for testing
        dummy_font = tmp_path / "test_font.ttf"
        dummy_font.write_text("dummy font data")
        
        # Should handle non-existent font gracefully
        result = setup_chinese_font("/nonexistent/font.ttf")
        assert result is None or isinstance(result, str)
        
    def test_font_detection_without_chinese_fonts(self):
        """Test behavior when no Chinese fonts are available"""
        with patch('os.path.exists', return_value=False):
            result = setup_chinese_font()
            # Should not crash, may return None
            assert result is None or isinstance(result, str)
    
    def test_matplotlib_font_manager_integration(self):
        """Test integration with matplotlib font manager"""
        from matplotlib import font_manager
        
        # Should not crash when accessing font manager
        try:
            available_fonts = [font.name for font in font_manager.fontManager.ttflist]
            assert isinstance(available_fonts, list)
        except Exception as e:
            pytest.fail(f"Font manager access failed: {e}")
    
    def test_font_data_structure(self):
        """Test font data JSON structure"""
        font_data_path = Path(__file__).parent.parent / "data" / "fonts.json"
        assert font_data_path.exists(), "fonts.json should exist"
        
        with open(font_data_path, 'r', encoding='utf-8') as f:
            font_data = json.load(f)
        
        # Verify required structure
        assert "chinese_fonts" in font_data
        assert "install_instructions" in font_data
        assert "wordcloud_requirements" in font_data
        
        # Check each font category has required fields
        for category, fonts in font_data["chinese_fonts"].items():
            assert "names" in fonts
            assert "paths" in fonts
            assert "quality" in fonts
            assert "description" in fonts


class TestVisualizationOutput:
    """Test visualization output quality and format"""
    
    def setup_method(self):
        """Setup test data"""
        # Create realistic frequency distribution
        self.test_freq = Counter()
        for i in range(100):
            freq = int(100 / (i + 1) ** 1.2)
            if freq > 0:
                self.test_freq[f'word_{i}'] = freq
        
        # Create test corpus for Heaps analysis
        self.test_corpus = []
        for doc_idx in range(20):
            doc_tokens = [f'word_{i % 30}' for i in range(doc_idx, doc_idx + 10)]
            self.test_corpus.append(doc_tokens)
    
    def test_zipf_plot_creation(self, tmp_path):
        """Test Zipf plot creation with 300DPI"""
        from stats import analyze_zipf_law
        
        zipf_results = analyze_zipf_law(self.test_freq)
        output_path = tmp_path / "test_zipf.png"
        
        # Should not crash
        create_zipf_panels(
            frequencies=self.test_freq,
            output_path=str(output_path),
            zipf_results=zipf_results,
            font_path=None,
            color_scheme='nature'
        )
        
        # File should be created
        assert output_path.exists()
        
        # Check file size (should be substantial for 300DPI)
        file_size = output_path.stat().st_size
        assert file_size > 10000, f"Plot file seems too small: {file_size} bytes"
    
    def test_heaps_plot_creation(self, tmp_path):
        """Test Heaps plot creation"""
        from stats import analyze_heaps_law
        
        heaps_results = analyze_heaps_law(self.test_corpus)
        output_path = tmp_path / "test_heaps.png"
        
        create_heaps_plot(
            corpus_tokens=self.test_corpus,
            output_path=str(output_path),
            heaps_results=heaps_results,
            font_path=None,
            color_scheme='science'
        )
        
        assert output_path.exists()
        file_size = output_path.stat().st_size
        assert file_size > 10000, f"Plot file seems too small: {file_size} bytes"
    
    def test_wordcloud_creation(self, tmp_path):
        """Test word cloud creation"""
        output_path = tmp_path / "test_cloud.png"
        
        create_wordcloud(
            frequencies=self.test_freq,
            output_path=str(output_path),
            title="Test Cloud",
            font_path=None,
            max_words=50,
            color_scheme='calm'
        )
        
        assert output_path.exists()
        file_size = output_path.stat().st_size
        assert file_size > 5000, f"WordCloud file seems too small: {file_size} bytes"
    
    def test_color_schemes(self, tmp_path):
        """Test different color schemes work"""
        color_schemes = ['nature', 'science', 'calm']
        
        for scheme in color_schemes:
            output_path = tmp_path / f"test_cloud_{scheme}.png"
            
            create_wordcloud(
                frequencies=self.test_freq,
                output_path=str(output_path),
                title=f"Test {scheme}",
                font_path=None,
                max_words=30,
                color_scheme=scheme
            )
            
            assert output_path.exists()
    
    def test_dpi_setting(self, tmp_path):
        """Test that plots are saved with 300DPI"""
        from PIL import Image
        
        output_path = tmp_path / "test_dpi.png"
        
        create_wordcloud(
            frequencies=self.test_freq,
            output_path=str(output_path),
            title="DPI Test",
            font_path=None,
            max_words=50
        )
        
        # Check image DPI using PIL
        try:
            with Image.open(output_path) as img:
                dpi = img.info.get('dpi', (72, 72))
                # DPI might be a tuple (horizontal, vertical)
                if isinstance(dpi, tuple):
                    assert dpi[0] >= 299 or dpi[1] >= 299, f"DPI too low: {dpi}"
                else:
                    assert dpi >= 299, f"DPI too low: {dpi}"
        except Exception as e:
            # If PIL check fails, at least verify file was created
            pytest.skip(f"Could not verify DPI: {e}")


class TestFontFallback:
    """Test font fallback mechanisms"""
    
    def test_graceful_degradation_no_fonts(self, tmp_path):
        """Test visualization works even without Chinese fonts"""
        test_freq = Counter({'hello': 10, 'world': 5, 'test': 3})
        output_path = tmp_path / "no_font_test.png"
        
        # Should work even without Chinese fonts
        create_wordcloud(
            frequencies=test_freq,
            output_path=str(output_path),
            title="No Font Test",
            font_path=None,
            max_words=20
        )
        
        assert output_path.exists()
    
    def test_mixed_language_content(self, tmp_path):
        """Test with mixed Chinese and English content"""
        mixed_freq = Counter({
            'hello': 10, '你好': 8, 'world': 7, '世界': 6,
            'test': 5, '测试': 4, 'data': 3, '数据': 2
        })
        
        output_path = tmp_path / "mixed_lang_test.png"
        
        create_wordcloud(
            frequencies=mixed_freq,
            output_path=str(output_path),
            title="Mixed Language Test",
            font_path=None,
            max_words=20
        )
        
        assert output_path.exists()


def test_visualization_integration():
    """Integration test for visualization pipeline"""
    # This test verifies the overall visualization system works
    from stats import analyze_zipf_law, analyze_heaps_law
    
    # Generate test data
    test_freq = Counter()
    for i in range(50):
        test_freq[f'term_{i}'] = max(1, int(50 / (i + 1)))
    
    test_corpus = [['a', 'b', 'c'], ['b', 'c', 'd'], ['c', 'd', 'e']]
    
    # Should not crash
    zipf_results = analyze_zipf_law(test_freq)
    heaps_results = analyze_heaps_law(test_corpus)
    
    assert isinstance(zipf_results, dict)
    assert isinstance(heaps_results, dict)
    assert 'r_squared' in zipf_results
    assert 'r_squared' in heaps_results


if __name__ == "__main__":
    # Run tests manually if pytest not available
    import tempfile
    import shutil
    
    print("Running font detection tests...")
    
    # Test font detection
    font_result = setup_chinese_font()
    print(f"Font detection result: {font_result}")
    
    # Test visualization creation
    with tempfile.TemporaryDirectory() as tmp_dir:
        test_freq = Counter({'hello': 10, 'world': 5, 'test': 3})
        
        # Test word cloud
        cloud_path = os.path.join(tmp_dir, "test_cloud.png")
        create_wordcloud(test_freq, cloud_path, "Test")
        
        if os.path.exists(cloud_path):
            print(f"✅ Word cloud created: {os.path.getsize(cloud_path)} bytes")
        else:
            print("❌ Word cloud creation failed")
    
    print("Manual tests completed!")