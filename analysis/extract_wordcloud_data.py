#!/usr/bin/env python3
"""
Extract detailed word frequency data for interactive word cloud
This script extracts yearly word frequencies to support the interactive web interface
"""

import json
import os
import sys
from pathlib import Path
from typing import Dict, Any
from collections import Counter

# Add pipeline to path
sys.path.insert(0, str(Path(__file__).parent / "pipeline"))

from corpus_io import load_corpus, split_by_year, read_text
from tokenizer import MixedLanguageTokenizer
from stats import calculate_frequencies_by_year


def extract_wordcloud_data(
    corpus_root: str = "../Wechat-Backup/文不加点的张衔瑜",
    output_path: str = "out/wordcloud_data.json",
    max_words_per_year: int = 200,
    max_words_overall: int = 500
) -> Dict[str, Any]:
    """
    Extract word frequency data for interactive word cloud
    
    Args:
        corpus_root: Path to corpus root
        output_path: Output JSON file path
        max_words_per_year: Maximum words per year to include
        max_words_overall: Maximum overall words to include
        
    Returns:
        Dict containing word frequency data
    """
    print("📊 Extracting word frequency data for interactive word cloud...")
    
    # Load corpus
    print("📁 Loading corpus...")
    corpus = load_corpus(corpus_root)
    if not corpus:
        print("❌ No corpus data found")
        return {}
    
    print(f"📈 Loaded {len(corpus)} documents")
    
    # Initialize tokenizer
    print("🔤 Initializing tokenizer...")
    tokenizer = MixedLanguageTokenizer(
        stopwords_zh_path="data/stopwords.zh.txt",
        stopwords_en_path="data/stopwords.en.txt",
        allow_singletons_path="data/allow_singletons.zh.txt"
    )
    
    # Split by year and tokenize
    print("📅 Splitting corpus by year and tokenizing...")
    corpus_by_year_articles = split_by_year(corpus)
    
    # Tokenize articles by year
    corpus_by_year = {}
    for year, articles in corpus_by_year_articles.items():
        print(f"   Processing year {year}: {len(articles)} articles")
        year_tokens = []
        for article in articles:
            content = read_text(article)
            if content.strip():
                tokens = tokenizer.tokenize(content)
                if tokens:  # Only add non-empty token lists
                    year_tokens.append(tokens)
        
        if year_tokens:  # Only add years with content
            corpus_by_year[year] = year_tokens
            print(f"   Year {year}: {len(year_tokens)} documents tokenized")
    
    if not corpus_by_year:
        print("❌ No tokenized content found")
        return {}
    
    # Calculate frequencies
    print("🔍 Calculating word frequencies...")
    freq_by_year = calculate_frequencies_by_year(corpus_by_year)
    
    # Calculate overall frequencies
    all_tokens = []
    for tokens_list in corpus_by_year.values():
        for tokens in tokens_list:
            all_tokens.extend(tokens)
    
    freq_overall = Counter(all_tokens)
    
    # Prepare data for web interface
    wordcloud_data = {
        "overall": {
            "words": [
                {"text": word, "size": freq} 
                for word, freq in freq_overall.most_common(max_words_overall)
            ],
            "total_words": sum(freq_overall.values()),
            "unique_words": len(freq_overall)
        },
        "by_year": {},
        "years": sorted(corpus_by_year.keys()),
        "metadata": {
            "extraction_timestamp": Path(__file__).stat().st_mtime,
            "max_words_per_year": max_words_per_year,
            "max_words_overall": max_words_overall,
            "total_documents": len(corpus)
        }
    }
    
    # Add yearly data
    for year in sorted(corpus_by_year.keys()):
        year_freq = freq_by_year.get(year, Counter())
        wordcloud_data["by_year"][year] = {
            "words": [
                {"text": word, "size": freq} 
                for word, freq in year_freq.most_common(max_words_per_year)
            ],
            "total_words": sum(year_freq.values()),
            "unique_words": len(year_freq),
            "documents": len(corpus_by_year[year])
        }
    
    # Save data
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(wordcloud_data, f, ensure_ascii=False, indent=2)
    
    print(f"💾 Word cloud data saved to: {output_path}")
    print(f"📊 Years available: {wordcloud_data['years']}")
    print(f"📈 Overall words: {len(wordcloud_data['overall']['words'])}")
    
    return wordcloud_data


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Extract word frequency data for interactive word cloud")
    parser.add_argument("--corpus-root", default="../Wechat-Backup/文不加点的张衔瑜", 
                      help="Path to corpus root directory")
    parser.add_argument("--output", default="out/wordcloud_data.json", 
                      help="Output JSON file path")
    parser.add_argument("--max-words-year", type=int, default=200,
                      help="Maximum words per year")
    parser.add_argument("--max-words-overall", type=int, default=500,
                      help="Maximum overall words")
    
    args = parser.parse_args()
    
    try:
        extract_wordcloud_data(
            corpus_root=args.corpus_root,
            output_path=args.output,
            max_words_per_year=args.max_words_year,
            max_words_overall=args.max_words_overall
        )
        print("✅ Data extraction completed successfully!")
    except Exception as e:
        print(f"❌ Error during extraction: {e}")
        sys.exit(1)