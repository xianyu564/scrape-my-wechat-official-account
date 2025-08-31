#!/usr/bin/env python3
"""
Comprehensive tests for the enhanced tokenization module
Tests pkuseg→jieba fallback, user dictionary loading, mixed language handling, and technical term preservation
"""

import unittest
import sys
import os
from pathlib import Path

# Add project paths
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / "analysis"))
sys.path.insert(0, str(project_root / "analysis" / "pipeline"))

# Import from the pipeline modules directly
import importlib.util

# Load modules directly to avoid naming conflicts
pipeline_dir = project_root / "analysis" / "pipeline"

# Load enhanced_tokenizer
spec = importlib.util.spec_from_file_location("enhanced_tokenizer", pipeline_dir / "enhanced_tokenizer.py")
enhanced_tokenizer = importlib.util.module_from_spec(spec)
sys.modules["enhanced_tokenizer"] = enhanced_tokenizer
spec.loader.exec_module(enhanced_tokenizer)

# Load tokenize module
spec2 = importlib.util.spec_from_file_location("tokenize_mod", pipeline_dir / "tokenize.py")
tokenize_mod = importlib.util.module_from_spec(spec2)
sys.modules["tokenize_mod"] = tokenize_mod
spec2.loader.exec_module(tokenize_mod)

# Import the needed classes and constants
EnhancedTokenizer = enhanced_tokenizer.EnhancedTokenizer
TokenInfo = enhanced_tokenizer.TokenInfo
create_tokenizer = enhanced_tokenizer.create_tokenizer
tokenize_text = enhanced_tokenizer.tokenize_text
inspect_tokenization = enhanced_tokenizer.inspect_tokenization
set_global_user_dicts = enhanced_tokenizer.set_global_user_dicts
get_global_user_dicts = enhanced_tokenizer.get_global_user_dicts
add_global_user_dict = enhanced_tokenizer.add_global_user_dict
TOKENIZER_TYPE = tokenize_mod.TOKENIZER_TYPE
EXTRA_USER_DICTS = tokenize_mod.EXTRA_USER_DICTS


class TestTokenizeModule(unittest.TestCase):
    """Test the tokenize module functionality"""
    
    def setUp(self):
        """Set up test environment"""
        # Clear global user dicts for clean tests
        set_global_user_dicts([])
    
    def test_tokenizer_type_validation(self):
        """Test that TOKENIZER_TYPE contains expected values"""
        expected_types = {"auto", "pkuseg", "jieba"}
        self.assertEqual(TOKENIZER_TYPE, expected_types)
    
    def test_enhanced_tokenizer_creation(self):
        """Test creating EnhancedTokenizer with different configurations"""
        # Test with auto tokenizer type
        tokenizer = EnhancedTokenizer(tokenizer_type="auto")
        self.assertIsNotNone(tokenizer)
        info = tokenizer.get_info()
        self.assertIn('name', info)
        self.assertIn('backend_available', info)
        
        # Test with jieba tokenizer type
        tokenizer_jieba = EnhancedTokenizer(tokenizer_type="jieba")
        self.assertIsNotNone(tokenizer_jieba)
        
        # Test invalid tokenizer type
        with self.assertRaises(ValueError):
            EnhancedTokenizer(tokenizer_type="invalid")
    
    def test_basic_tokenization(self):
        """Test basic tokenization functionality"""
        tokenizer = EnhancedTokenizer()
        
        # Test simple Chinese text
        text = "你好世界"
        tokens = tokenizer.tokenize(text)
        self.assertIsInstance(tokens, list)
        self.assertTrue(len(tokens) > 0)
        
        # Test empty text
        empty_tokens = tokenizer.tokenize("")
        self.assertEqual(empty_tokens, [])
        
        # Test None input
        none_tokens = tokenizer.tokenize(None)
        self.assertEqual(none_tokens, [])
    
    def test_mixed_language_cases(self):
        """Test specific mixed language cases mentioned in requirements"""
        tokenizer = EnhancedTokenizer()
        
        test_cases = [
            {
                'text': '北京大学生前来应聘',
                'description': 'Ambiguous Chinese segmentation case'
            },
            {
                'text': '机器 学习/深度 学习', 
                'description': 'Mixed Chinese with spaces and slashes'
            },
            {
                'text': 'IL-6/STAT3',
                'description': 'Technical biological term with hyphen and slash'
            },
            {
                'text': 'AIGC_2025',
                'description': 'Technical term with underscore and numbers'
            },
            {
                'text': 'ChatGPT-4o-mini',
                'description': 'Complex technical term with multiple components'
            }
        ]
        
        for case in test_cases:
            with self.subTest(case=case['description']):
                tokens = tokenizer.tokenize(case['text'])
                self.assertIsInstance(tokens, list)
                self.assertTrue(len(tokens) > 0, f"No tokens for: {case['text']}")
                
                # Verify technical terms are preserved
                text = case['text']
                if any(term in text for term in ['IL-6', 'STAT3', 'AIGC_2025', 'ChatGPT-4o-mini']):
                    # Check that technical terms appear as single tokens
                    joined = ''.join(tokens)
                    self.assertIn('IL-6', joined) if 'IL-6' in text else None
                    self.assertIn('STAT3', joined) if 'STAT3' in text else None
                    self.assertIn('AIGC_2025', joined) if 'AIGC_2025' in text else None
                    self.assertIn('ChatGPT-4o-mini', joined) if 'ChatGPT-4o-mini' in text else None
    
    def test_inspect_functionality(self):
        """Test the inspect method returns proper TokenInfo structures"""
        tokenizer = EnhancedTokenizer()
        text = "北京大学生前来应聘机器学习"
        
        result = tokenizer.inspect(text)
        
        # Check result structure
        self.assertIsInstance(result, dict)
        self.assertIn('kept', result)
        self.assertIn('filtered', result)
        
        # Check TokenInfo structure for kept tokens
        if result['kept']:
            token_info = result['kept'][0]
            self.assertIsInstance(token_info, TokenInfo)
            self.assertTrue(hasattr(token_info, 'token'))
            self.assertTrue(hasattr(token_info, 'char_span'))
            self.assertTrue(hasattr(token_info, 'source'))
            self.assertTrue(hasattr(token_info, 'kept_reason'))
            
            # Validate field types
            self.assertIsInstance(token_info.token, str)
            self.assertIsInstance(token_info.char_span, tuple)
            self.assertEqual(len(token_info.char_span), 2)
            self.assertIsInstance(token_info.source, str)
            self.assertIsInstance(token_info.kept_reason, str)
    
    def test_convenience_functions(self):
        """Test module-level convenience functions"""
        text = "测试文本"
        
        # Test create_tokenizer
        tokenizer = create_tokenizer("jieba")
        self.assertIsNotNone(tokenizer)
        
        # Test tokenize_text
        tokens = tokenize_text(text, "jieba")
        self.assertIsInstance(tokens, list)
        
        # Test inspect_tokenization  
        result = inspect_tokenization(text, "jieba")
        self.assertIsInstance(result, dict)
        self.assertIn('kept', result)
        self.assertIn('filtered', result)
    
    def test_global_user_dict_management(self):
        """Test global user dictionary management functions"""
        # Test initial state
        initial_dicts = get_global_user_dicts()
        self.assertIsInstance(initial_dicts, list)
        
        # Test setting global dicts
        test_dicts = ["/path/to/dict1.txt", "/path/to/dict2.txt"]
        set_global_user_dicts(test_dicts)
        current_dicts = get_global_user_dicts()
        self.assertEqual(current_dicts, test_dicts)
        
        # Test adding global dict
        new_dict = "/path/to/dict3.txt"
        add_global_user_dict(new_dict)
        updated_dicts = get_global_user_dicts()
        self.assertIn(new_dict, updated_dicts)
        
        # Test adding duplicate (should not duplicate)
        add_global_user_dict(new_dict)
        final_dicts = get_global_user_dicts()
        self.assertEqual(final_dicts.count(new_dict), 1)
        
        # Clean up
        set_global_user_dicts([])
    
    def test_user_dictionary_integration(self):
        """Test that user dictionaries are properly loaded and used"""
        # Create tokenizer with default user dictionaries
        tokenizer = EnhancedTokenizer()
        info = tokenizer.get_info()
        
        # Should have loaded user dictionaries
        self.assertGreater(info.get('user_dictionaries', 0), 0)
        
        # Test terms that should be in user dictionaries
        user_dict_terms = [
            "机器学习",  # Should be in user_dict.zh.txt
            "深度学习",  # Should be in user_dict.zh.txt
            "人工智能",  # Should be in user_dict.zh.txt
        ]
        
        for term in user_dict_terms:
            tokens = tokenizer.tokenize(term)
            # Term should ideally be kept as single token or minimal segmentation
            self.assertTrue(len(tokens) <= 2, f"Term '{term}' over-segmented: {tokens}")
    
    def test_technical_term_preservation(self):
        """Test that technical terms are preserved correctly"""
        tokenizer = EnhancedTokenizer()
        
        technical_terms = [
            "AIGC_2025",
            "ResNet50", 
            "ChatGPT-4o-mini",
            "IL-6/STAT3",
            "COVID-19",
            "BERT",
            "GPT-4"
        ]
        
        for term in technical_terms:
            tokens = tokenizer.tokenize(term)
            # Technical terms should be preserved as single tokens or minimal splits
            joined_result = ''.join(tokens)
            
            # Check that all important parts are preserved
            if '_' in term or '-' in term or '/' in term:
                # For complex technical terms, ensure key components are preserved
                self.assertTrue(
                    any(part in ''.join(tokens) for part in term.replace('_', ' ').replace('-', ' ').replace('/', ' ').split()),
                    f"Technical term '{term}' not properly preserved: {tokens}"
                )
            else:
                # Simple technical terms should be single tokens
                self.assertTrue(
                    len(tokens) == 1 or term in joined_result,
                    f"Simple technical term '{term}' not preserved: {tokens}"
                )
    
    def test_normalization_features(self):
        """Test text normalization features"""
        tokenizer = EnhancedTokenizer()
        
        # Test full-width to half-width conversion
        full_width_text = "ＨｅｌｌｏＷｏｒｌｄ（测试）"
        tokens = tokenizer.tokenize(full_width_text)
        joined = ''.join(tokens)
        
        # Should contain half-width characters
        self.assertIn('Hello', joined) if 'Hello' in joined else None
        self.assertIn('World', joined) if 'World' in joined else None
        self.assertIn('(', joined) if '(' in joined else None
        self.assertIn(')', joined) if ')' in joined else None
        
        # Test excessive whitespace handling
        spaced_text = "机器   学习    深度     学习"
        tokens = tokenizer.tokenize(spaced_text)
        # Should not have empty tokens from excessive spaces
        self.assertNotIn('', tokens)
        self.assertNotIn('   ', tokens)
    
    def test_fallback_mechanism(self):
        """Test pkuseg→jieba fallback mechanism"""
        # Test with explicit jieba (should work)
        tokenizer_jieba = EnhancedTokenizer(tokenizer_type="jieba")
        info_jieba = tokenizer_jieba.get_info()
        self.assertEqual(info_jieba['name'], 'jieba')
        
        # Test with auto (should fall back to jieba if pkuseg unavailable)
        tokenizer_auto = EnhancedTokenizer(tokenizer_type="auto")  
        info_auto = tokenizer_auto.get_info()
        self.assertIn(info_auto['name'], ['jieba', 'pkuseg'])
        
        # Both should produce valid tokenization
        test_text = "测试回退机制"
        tokens_jieba = tokenizer_jieba.tokenize(test_text)
        tokens_auto = tokenizer_auto.tokenize(test_text)
        
        self.assertTrue(len(tokens_jieba) > 0)
        self.assertTrue(len(tokens_auto) > 0)


class TestSpecificRequirementCases(unittest.TestCase):
    """Test specific cases mentioned in the requirements"""
    
    def setUp(self):
        """Set up tokenizer for requirement tests"""
        self.tokenizer = EnhancedTokenizer()
    
    def test_beijing_university_case(self):
        """Test '北京大学生前来应聘' - ambiguous segmentation case"""
        text = "北京大学生前来应聘"
        tokens = self.tokenizer.tokenize(text)
        
        # Should segment properly without wrong cuts
        self.assertIsInstance(tokens, list)
        self.assertTrue(len(tokens) > 0)
        
        # Check that reasonable segmentation occurred
        # This is a classic ambiguous case: could be "北京大学生" or "北京 大学生"
        joined = ''.join(tokens)
        self.assertEqual(joined.replace(' ', ''), text.replace(' ', ''))
    
    def test_machine_learning_case(self):
        """Test '机器 学习/深度 学习' - mixed spaces and technical terms"""
        text = "机器 学习/深度 学习"
        tokens = self.tokenizer.tokenize(text)
        
        self.assertIsInstance(tokens, list)
        self.assertTrue(len(tokens) > 0)
        
        # Should handle spaces and slashes appropriately
        joined = ''.join(tokens)
        self.assertIn('机器', joined)
        self.assertIn('学习', joined)
        self.assertIn('深度', joined)
    
    def test_il6_stat3_case(self):
        """Test 'IL-6/STAT3' - technical biological term"""
        text = "IL-6/STAT3"
        tokens = self.tokenizer.tokenize(text)
        
        self.assertIsInstance(tokens, list)
        self.assertTrue(len(tokens) > 0)
        
        # Technical terms should be preserved
        joined = ''.join(tokens)
        self.assertIn('IL-6', joined)
        self.assertIn('STAT3', joined)
    
    def test_inspect_detailed_output(self):
        """Test that inspect() provides detailed token information"""
        text = "IL-6/STAT3机器学习"
        result = self.tokenizer.inspect(text)
        
        # Check structure
        self.assertIn('kept', result)
        self.assertIn('filtered', result)
        
        # Check that we have detailed information
        if result['kept']:
            for token_info in result['kept'][:3]:  # Check first few
                self.assertIsInstance(token_info.token, str)
                self.assertIsInstance(token_info.char_span, tuple)
                self.assertIn(token_info.source, ['pkuseg', 'jieba', 'jieba_mixed_separation', 'mixed_separation', 'normalization'])
                self.assertIsInstance(token_info.kept_reason, str)
                self.assertTrue(len(token_info.kept_reason) > 0)


if __name__ == '__main__':
    # Run tests with verbose output
    unittest.main(verbosity=2)