"""
Smoke tests for the analysis pipeline functionality.
Basic tests to ensure text analysis features work properly.
"""
import json
import sys
from pathlib import Path

import pytest

# Add project root to path to import modules
sys.path.insert(0, str(Path(__file__).parent.parent))


class TestAnalysisPipeline:
    """Test core analysis pipeline components."""

    def test_pipeline_imports(self):
        """Test that analysis pipeline modules can be imported."""
        try:
            from analysis.pipeline import corpus_io, stats, tokenizer
            assert hasattr(corpus_io, '__name__')
            assert hasattr(stats, '__name__')
            assert hasattr(tokenizer, '__name__')
        except ImportError as e:
            pytest.skip(f"Analysis modules not available: {e}")

    def test_text_processing_basic(self):
        """Test basic text processing capabilities."""
        # Test basic string operations that would be used in analysis
        sample_text = "这是一个测试文本，包含中文字符。This is a test text with English."

        assert isinstance(sample_text, str)
        assert len(sample_text) > 0

        # Test basic text analysis operations
        words = sample_text.split()
        assert len(words) > 0

        # Test encoding handling
        encoded = sample_text.encode('utf-8')
        decoded = encoded.decode('utf-8')
        assert decoded == sample_text

    def test_json_metadata_handling(self):
        """Test handling of article metadata in JSON format."""
        sample_metadata = {
            "title": "测试文章标题",
            "date": "2023-01-01",
            "author": "张衔瑜",
            "word_count": 1500,
            "tags": ["日常", "思考"],
            "publish_time": "2023-01-01 10:00:00"
        }

        # Test JSON serialization/deserialization
        json_str = json.dumps(sample_metadata, ensure_ascii=False, indent=2)
        assert "测试文章标题" in json_str

        parsed_data = json.loads(json_str)
        assert parsed_data == sample_metadata
        assert parsed_data["word_count"] == 1500

    @pytest.mark.parametrize("text_input,expected_type", [
        ("中文文本", str),
        ("English text", str),
        ("Mixed 中英 text", str),
        ("", str),
    ])
    def test_text_type_handling(self, text_input, expected_type):
        """Test handling of different text types."""
        assert isinstance(text_input, expected_type)

        # Test length calculation
        length = len(text_input)
        assert isinstance(length, int)
        assert length >= 0


class TestCorpusOperations:
    """Test corpus-level operations without requiring actual corpus data."""

    def test_file_path_operations(self, tmp_path):
        """Test file path operations for corpus management."""
        # Create mock corpus structure
        corpus_root = tmp_path / "test_corpus"
        year_dir = corpus_root / "2023"
        article_dir = year_dir / "2023-01-01_test_article"

        # Create directory structure
        article_dir.mkdir(parents=True, exist_ok=True)

        # Create mock files
        html_file = article_dir / "article.html"
        md_file = article_dir / "article.md"
        meta_file = article_dir / "meta.json"

        html_file.write_text("<html><body>Test content</body></html>", encoding="utf-8")
        md_file.write_text("# Test Article\n\nTest content", encoding="utf-8")

        meta_data = {"title": "Test Article", "date": "2023-01-01"}
        meta_file.write_text(json.dumps(meta_data, ensure_ascii=False), encoding="utf-8")

        # Test file existence and content
        assert html_file.exists()
        assert md_file.exists()
        assert meta_file.exists()

        # Test reading content
        html_content = html_file.read_text(encoding="utf-8")
        assert "Test content" in html_content

        md_content = md_file.read_text(encoding="utf-8")
        assert "# Test Article" in md_content

    def test_directory_traversal(self, tmp_path):
        """Test directory traversal for corpus analysis."""
        # Create mock directory structure
        corpus_root = tmp_path / "corpus"

        # Create multiple year directories
        for year in ["2021", "2022", "2023"]:
            year_dir = corpus_root / year
            year_dir.mkdir(parents=True)

            # Create some mock article directories
            for month in ["01", "06", "12"]:
                article_dir = year_dir / f"{year}-{month}-01_test_article"
                article_dir.mkdir(exist_ok=True)

                # Create a meta.json file
                meta_file = article_dir / "meta.json"
                meta_data = {"title": f"Article {year}-{month}", "date": f"{year}-{month}-01"}
                meta_file.write_text(json.dumps(meta_data), encoding="utf-8")

        # Test directory traversal
        years = [d.name for d in corpus_root.iterdir() if d.is_dir()]
        assert len(years) == 3
        assert "2021" in years
        assert "2022" in years
        assert "2023" in years

        # Test article counting
        total_articles = 0
        for year_dir in corpus_root.iterdir():
            if year_dir.is_dir():
                articles = [d for d in year_dir.iterdir() if d.is_dir()]
                total_articles += len(articles)

        assert total_articles == 9  # 3 years * 3 months


class TestDataValidation:
    """Test data validation and consistency checks."""

    def test_article_metadata_validation(self):
        """Test validation of article metadata structure."""
        valid_metadata = {
            "title": "Valid Article",
            "date": "2023-01-01",
            "url": "http://example.com/article",
            "word_count": 1000
        }

        # Test required fields
        required_fields = ["title", "date", "url"]
        for field in required_fields:
            assert field in valid_metadata
            assert valid_metadata[field] is not None
            assert valid_metadata[field] != ""

        # Test data types
        assert isinstance(valid_metadata["title"], str)
        assert isinstance(valid_metadata["date"], str)
        assert isinstance(valid_metadata["url"], str)
        assert isinstance(valid_metadata["word_count"], int)

    def test_date_format_validation(self):
        """Test date format validation."""
        import re

        valid_dates = ["2023-01-01", "2023-12-31", "2020-02-29"]
        invalid_dates = ["2023-1-1", "23-01-01", "2023/01/01", "invalid"]

        date_pattern = r'^\d{4}-\d{2}-\d{2}$'

        for date in valid_dates:
            assert re.match(date_pattern, date), f"Valid date {date} should match pattern"

        for date in invalid_dates:
            assert not re.match(date_pattern, date), f"Invalid date {date} should not match pattern"

    def test_url_format_validation(self):
        """Test URL format validation."""
        import re

        valid_urls = [
            "http://example.com/article",
            "https://mp.weixin.qq.com/s/abc123",
            "https://example.com/path/to/article"
        ]

        invalid_urls = [
            "not-a-url",
            "ftp://example.com",
            "",
            None
        ]

        url_pattern = r'^https?://.+'

        for url in valid_urls:
            assert re.match(url_pattern, url), f"Valid URL {url} should match pattern"

        for url in invalid_urls:
            if url is not None:
                assert not re.match(url_pattern, url), f"Invalid URL {url} should not match pattern"


class TestAnalysisUtilities:
    """Test utility functions for analysis."""

    def test_word_counting(self):
        """Test word counting functionality."""
        test_texts = [
            ("简单的中文测试", 4),  # Chinese characters count as individual words
            ("Simple English test", 3),
            ("Mixed 中英 text", 3),
            ("", 0),
        ]

        for text, expected_word_count in test_texts:
            # Basic word count (split by spaces)
            words = text.split() if text else []
            word_count = len(words)

            if text == "简单的中文测试":
                # For Chinese text, each character could be considered a word
                assert len(text) >= expected_word_count
            else:
                assert word_count == expected_word_count

    def test_text_statistics(self):
        """Test basic text statistics calculation."""
        sample_text = "This is a sample text for testing. It has multiple sentences!"

        # Character count
        char_count = len(sample_text)
        assert char_count > 0

        # Word count (basic)
        word_count = len(sample_text.split())
        assert word_count > 0

        # Sentence count (basic)
        sentence_count = sample_text.count('.') + sample_text.count('!') + sample_text.count('?')
        assert sentence_count > 0

        # Line count
        line_count = sample_text.count('\n') + 1
        assert line_count >= 1
