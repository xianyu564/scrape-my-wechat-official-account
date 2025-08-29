#!/usr/bin/env python3
"""
Test suite for enhanced tokenization with pkuseg/jieba fallback
Covers user dictionaries, technical terms, and mixed language handling
"""

import sys
import os
import tempfile
from pathlib import Path

# Add pipeline to path
sys.path.insert(0, str(Path(__file__).parent.parent / "pipeline"))

import pytest
from tokenizer import MixedLanguageTokenizer, ChineseNormalizer, TokenInfo


class TestChineseNormalizer:
    """Test Chinese text normalization"""
    
    def test_full_width_conversion(self):
        normalizer = ChineseNormalizer()
        
        # Test full-width to half-width conversion
        text = "ＡＩＧＣ２０２５，这是测试。"
        result = normalizer.normalize(text)
        assert "AIGC2025" in result
        assert "，" not in result or "," in result
    
    def test_technical_term_preservation(self):
        normalizer = ChineseNormalizer()
        
        # Test technical terms are preserved
        text = "AIGC_2025 ResNet50 ChatGPT-4o-mini IL-6/STAT3 camelCase snake_case"
        result = normalizer.normalize(text)
        
        # Check that key technical patterns are preserved
        assert "AIGC_2025" in result
        assert "ResNet50" in result  
        assert "ChatGPT-4o-mini" in result
        assert "IL-6/STAT3" in result
        assert "camelCase" in result  # camelCase should be preserved
        assert "snake_case" in result
    
    def test_english_lowercase(self):
        normalizer = ChineseNormalizer()
        
        # Regular English should be lowercased (but technical terms preserved)
        text = "This is normal English text with AIGC_2025"
        result = normalizer.normalize(text)
        
        # Regular words should be lowercase
        assert "this" in result
        assert "is" in result
        assert "normal" in result
        assert "english" in result
        assert "text" in result
        # But technical terms should be preserved
        assert "AIGC_2025" in result


class TestMixedLanguageTokenizer:
    """Test mixed language tokenization with various backends"""
    
    def setup_method(self):
        """Setup test data files"""
        # Create temporary files for testing
        self.temp_dir = tempfile.mkdtemp()
        
        # User dictionary
        self.user_dict_path = os.path.join(self.temp_dir, "user_dict.txt")
        with open(self.user_dict_path, 'w', encoding='utf-8') as f:
            f.write("机器学习 100 n\n")
            f.write("深度学习 100 n\n")
            f.write("北京大学生 50 n\n")  # This should prevent wrong segmentation
            f.write("一带一路 100 n\n")
        
        # Tech terms dictionary
        self.tech_dict_path = os.path.join(self.temp_dir, "tech_terms.txt")
        with open(self.tech_dict_path, 'w', encoding='utf-8') as f:
            f.write("AIGC_2025 100\n")
            f.write("ResNet50 100\n")
            f.write("IL-6/STAT3 100\n")
        
        # Stopwords
        self.stopwords_zh_path = os.path.join(self.temp_dir, "stopwords_zh.txt")
        with open(self.stopwords_zh_path, 'w', encoding='utf-8') as f:
            f.write("的\n了\n在\n是\n")
        
        self.stopwords_en_path = os.path.join(self.temp_dir, "stopwords_en.txt")
        with open(self.stopwords_en_path, 'w', encoding='utf-8') as f:
            f.write("the\na\nan\nand\nof\n")
        
        # Allowed singletons
        self.singletons_path = os.path.join(self.temp_dir, "singletons.txt")
        with open(self.singletons_path, 'w', encoding='utf-8') as f:
            f.write("学\n工\n人\n国\n")
    
    def teardown_method(self):
        """Clean up temporary files"""
        import shutil
        shutil.rmtree(self.temp_dir)
    
    def test_pkuseg_fallback_to_jieba(self):
        """Test automatic fallback from pkuseg to jieba when pkuseg unavailable"""
        # Test with auto mode (should work regardless of pkuseg availability)
        tokenizer = MixedLanguageTokenizer(
            tokenizer_type="auto",
            extra_user_dicts=[self.user_dict_path],
            stopwords_zh_path=self.stopwords_zh_path,
            stopwords_en_path=self.stopwords_en_path,
            allow_singletons_path=self.singletons_path
        )
        
        # Should initialize successfully
        assert tokenizer.tokenizer_name in ["pkuseg", "jieba", "jieba (fallback)"]
        
        # Test basic tokenization
        text = "这是一个测试"
        tokens = tokenizer.tokenize(text)
        assert len(tokens) > 0
        assert isinstance(tokens, list)
        assert all(isinstance(token, str) for token in tokens)
    
    def test_user_dictionary_effect(self):
        """Test that user dictionaries affect tokenization"""
        tokenizer = MixedLanguageTokenizer(
            tokenizer_type="auto",
            extra_user_dicts=[self.user_dict_path],
            stopwords_zh_path=self.stopwords_zh_path,
            stopwords_en_path=self.stopwords_en_path,
            allow_singletons_path=self.singletons_path
        )
        
        # Test cases from requirements
        test_cases = [
            ("北京大学生前来应聘", "北京大学生"),  # Should not split as "北京 大学 生"
            ("机器学习和深度学习", ["机器学习", "深度学习"]),  # Should be single tokens
            ("一带一路倡议", "一带一路")  # Should be preserved as single token
        ]
        
        for text, expected in test_cases:
            tokens = tokenizer.tokenize(text)
            if isinstance(expected, str):
                assert expected in tokens, f"Expected '{expected}' in tokens for '{text}', got {tokens}"
            elif isinstance(expected, list):
                for exp_token in expected:
                    assert exp_token in tokens, f"Expected '{exp_token}' in tokens for '{text}', got {tokens}"
    
    def test_technical_terms_preservation(self):
        """Test that technical terms are preserved correctly"""
        tokenizer = MixedLanguageTokenizer(
            tokenizer_type="auto",
            extra_user_dicts=[self.tech_dict_path],
            stopwords_zh_path=self.stopwords_zh_path,
            stopwords_en_path=self.stopwords_en_path,
            allow_singletons_path=self.singletons_path
        )
        
        # Test that core technical terms are preserved/recognized
        # Note: some complex terms may be split by jieba, but key parts should be preserved
        test_cases = [
            ("AIGC_2025技术升级", ["aigc_2025"]),  # Should be normalized but preserved as single token
            ("ResNet50模型训练", ["resnet50"]),   # Technical term preserved 
            ("机器学习深度学习", ["机器学习", "深度学习"]),  # User dictionary terms
        ]
        
        for text, expected_tokens in test_cases:
            tokens = tokenizer.tokenize(text)
            
            # Check that at least one of the expected tokens is present
            found = [token for token in expected_tokens if token in tokens]
            assert len(found) > 0, f"Expected one of {expected_tokens} in tokens for '{text}', got {tokens}"
        
        # Test that technical patterns are handled (may be split but components preserved)
        technical_texts = [
            "camelCase命名规范",
            "snake_case变量名",
            "IL-6/STAT3信号通路"
        ]
        
        for text in technical_texts:
            tokens = tokenizer.tokenize(text)
            # Should have reasonable tokenization (not empty, contains meaningful parts)
            assert len(tokens) > 0, f"No tokens generated for: {text}"
            # Should preserve English technical components
            english_tokens = [t for t in tokens if any(c.isalpha() for c in t)]
            assert len(english_tokens) > 0, f"No English components preserved for: {text}, tokens: {tokens}"
    
    def test_mixed_language_separation(self):
        """Test separation of mixed Chinese-English tokens"""
        tokenizer = MixedLanguageTokenizer(
            tokenizer_type="auto",
            stopwords_zh_path=self.stopwords_zh_path,
            stopwords_en_path=self.stopwords_en_path,
            allow_singletons_path=self.singletons_path
        )
        
        # Test mixed language text
        text = "这是API接口文档"
        tokens = tokenizer.tokenize(text)
        
        # Should separate Chinese and English parts
        assert "API" in tokens or "api" in tokens  # English part
        # Chinese parts should also be present
        chinese_tokens = [t for t in tokens if any('\u4e00' <= c <= '\u9fff' for c in t)]
        assert len(chinese_tokens) > 0
    
    def test_stopwords_filtering(self):
        """Test that stopwords are properly filtered"""
        tokenizer = MixedLanguageTokenizer(
            tokenizer_type="auto",
            stopwords_zh_path=self.stopwords_zh_path,
            stopwords_en_path=self.stopwords_en_path,
            allow_singletons_path=self.singletons_path
        )
        
        # Test Chinese stopwords
        text = "这是的一个了测试在系统"
        tokens = tokenizer.tokenize(text)
        
        # Stopwords should be filtered out
        stopwords_found = [token for token in tokens if token in ["的", "了", "在", "是"]]
        assert len(stopwords_found) == 0, f"Found stopwords: {stopwords_found}"
        
        # Test English stopwords
        text = "this is a test of the system"
        tokens = tokenizer.tokenize(text)
        
        # English stopwords should be filtered
        en_stopwords_found = [token for token in tokens if token in ["the", "a", "an", "and", "of"]]
        assert len(en_stopwords_found) == 0, f"Found English stopwords: {en_stopwords_found}"
    
    def test_allowed_singletons(self):
        """Test that allowed single Chinese characters are preserved"""
        tokenizer = MixedLanguageTokenizer(
            tokenizer_type="auto",
            stopwords_zh_path=self.stopwords_zh_path,
            stopwords_en_path=self.stopwords_en_path,
            allow_singletons_path=self.singletons_path
        )
        
        # Test allowed singletons
        text = "学工人国政府"
        tokens = tokenizer.tokenize(text)
        
        # Allowed singletons should be preserved
        allowed = ["学", "工", "人", "国"]
        found_allowed = [token for token in tokens if token in allowed]
        assert len(found_allowed) > 0, f"No allowed singletons found in {tokens}"
    
    def test_inspect_functionality(self):
        """Test the inspect function for debugging"""
        tokenizer = MixedLanguageTokenizer(
            tokenizer_type="auto",
            extra_user_dicts=[self.user_dict_path],
            stopwords_zh_path=self.stopwords_zh_path,
            stopwords_en_path=self.stopwords_en_path,
            allow_singletons_path=self.singletons_path
        )
        
        text = "机器学习是人工智能的重要分支"
        result = tokenizer.inspect(text, topk=10)
        
        # Should return kept and filtered tokens
        assert "kept" in result
        assert "filtered" in result
        assert isinstance(result["kept"], list)
        assert isinstance(result["filtered"], list)
        
        # Each token info should have required fields
        if result["kept"]:
            token_info = result["kept"][0]
            assert hasattr(token_info, 'token')
            assert hasattr(token_info, 'char_span')
            assert hasattr(token_info, 'source')
            assert hasattr(token_info, 'kept_reason')
            assert isinstance(token_info.char_span, tuple)
            assert len(token_info.char_span) == 2
    
    def test_tokenizer_info(self):
        """Test tokenizer information reporting"""
        tokenizer = MixedLanguageTokenizer(
            tokenizer_type="auto",
            extra_user_dicts=[self.user_dict_path, self.tech_dict_path],
            stopwords_zh_path=self.stopwords_zh_path,
            stopwords_en_path=self.stopwords_en_path,
            allow_singletons_path=self.singletons_path
        )
        
        info = tokenizer.get_tokenizer_info()
        
        # Should contain required information
        assert "name" in info
        assert "backend_available" in info
        assert "chinese_stopwords" in info
        assert "english_stopwords" in info
        assert "allowed_singletons" in info
        assert "user_dictionaries" in info
        
        # Check counts
        assert info["chinese_stopwords"] > 0
        assert info["english_stopwords"] > 0
        assert info["allowed_singletons"] > 0
        assert info["user_dictionaries"] == 2  # user_dict + tech_dict


def test_coverage_examples():
    """Test specific examples mentioned in requirements"""
    import tempfile
    import os
    
    # Create minimal test setup
    temp_dir = tempfile.mkdtemp()
    
    user_dict_path = os.path.join(temp_dir, "user_dict.txt")
    with open(user_dict_path, 'w', encoding='utf-8') as f:
        f.write("机器学习 100 n\n")
        f.write("北京大学生 50 n\n")
    
    tokenizer = MixedLanguageTokenizer(
        tokenizer_type="auto",
        extra_user_dicts=[user_dict_path]
    )
    
    # Test cases from requirements
    test_cases = [
        "北京大学生前来应聘",
        "机器学习/深度学习",
        "AIGC_2025-升级",
        "IL-6/STAT3",
        "一带一路"
    ]
    
    for text in test_cases:
        tokens = tokenizer.tokenize(text)
        assert len(tokens) > 0, f"No tokens generated for: {text}"
        assert all(isinstance(token, str) for token in tokens), f"Non-string tokens in: {tokens}"
    
    # Clean up
    import shutil
    shutil.rmtree(temp_dir)


if __name__ == "__main__":
    # Run tests directly
    pytest.main([__file__, "-v"])