# 📊 Robust Chinese + Mixed-Language Linguistic Analysis Implementation

This implementation provides a comprehensive linguistic analysis system with variable-length n-grams, advanced tokenization, and scientific metrics, designed specifically for Chinese and mixed-language content analysis.

## 🏗️ Architecture

### Two-Phase Design
1. **Phase 1: Theory-Only Analysis** - Pure computational linguistics without visualization
2. **Phase 2: Presentation** - Beautiful visualizations and reports

### Directory Structure
```
analysis/
├── main.py                    # Central control with all configuration knobs
├── pipeline/                  # Modular analysis components
│   ├── corpus_io.py          # Corpus loading and year-based splitting
│   ├── tokenize.py           # Pluggable tokenizers (pkuseg → jieba)
│   ├── ngrams.py             # Variable-length n-gram builder + collocation
│   ├── stats.py              # Scientific metrics (Zipf/Heaps/TF-IDF)
│   ├── viz.py                # Scientific-grade visualizations
│   └── report.py             # Markdown report generation
├── data/                     # Linguistic resources
│   ├── stopwords.zh.txt      # Chinese stopwords
│   ├── stopwords.en.txt      # English stopwords
│   └── allow_singletons.zh.txt # Meaningful single Chinese characters
└── out/                      # Single output sink
```

## ✨ Key Features

### 🔤 Advanced Tokenization
- **Primary**: pkuseg (domain-general model for robust Chinese segmentation)
- **Fallback**: jieba (if pkuseg unavailable)
- **Normalization**: Full-width → half-width, English lowercase, preserve technical terms
- **Mixed Language**: Intelligent handling of machine-learning, AIGC_2025, etc.
- **Singleton Filtering**: Meaningful Chinese single characters preserved (人/心/光)

### 🔢 Variable-Length N-grams
- **Not capped at 1–4**: Supports 1-8+ grams for complex linguistic structures
- **Collocation Filtering**: PMI or log-likelihood ratio for phrase validation
- **Semantic Merging**: Validated phrases become single tokens (机器_学习)
- **Self-Validation**: Ensures ngram_lengths_detected includes all expected lengths

### 📈 Scientific Metrics
- **Zipf's Law**: Rank-frequency analysis with R² validation
- **Heaps' Law**: Vocabulary growth curves (V = K × n^β)
- **TF-IDF**: scikit-learn integration with pre-tokenized input
- **Lexical Diversity**: TTR, lexical density, content word ratios
- **YoY Growth**: Year-over-year word frequency changes

### 🎨 Beautiful Visualizations
- **Word Clouds**: CJK font support, scientific color schemes, 300 DPI
- **Scientific Plots**: Four-panel Zipf analysis, Heaps curves with confidence intervals
- **Comparative Charts**: Yearly frequency comparisons, growth visualizations
- **Reproducible**: Fixed random seeds for consistent results

### 📝 Clean Report Generation
- **Cheer-Up Tone**: Uplifting language designed to "cheer myself up from my past papers"
- **Scientific Structure**: Executive summary, methods, global overview, yearly snapshots
- **Phrase Inventory**: N-gram distribution analysis with examples
- **Reproducibility**: Complete parameter documentation for replication

## ⚙️ Configuration (in main.py)

### Execution Control
```python
RUN_ANALYSIS = True          # Phase 1: Theory-only analysis
RUN_VISUALIZATION = True     # Phase 2: Presentation
```

### N-gram Parameters
```python
MAX_N = 8                    # Maximum n-gram length (not capped!)
MIN_FREQ = 5                 # Minimum frequency for retention
COLLOCATION = "pmi"          # "pmi" or "llr" for collocation filtering
PMI_THRESHOLD = 3.0          # PMI threshold for phrase validation
```

### Tokenization
```python
TOKENIZER_TYPE = "auto"      # "pkuseg", "jieba", or "auto"
```

### Visualization
```python
FONT_PATH = None             # Auto-detect Chinese fonts
COLOR_SCHEME = "nature"      # Scientific color schemes
```

## 🚀 Usage

```bash
# Full analysis (both phases)
cd analysis/
python main.py

# Phase 1 only (set RUN_VISUALIZATION = False)
# Phase 2 only (set RUN_ANALYSIS = False, requires previous phase 1 results)
```

## 📊 Output Files

### Analysis Results
- `out/summary.json` - Comprehensive statistics summary
- `out/analysis_results.pkl` - Complete analysis data for phase 2

### Scientific Plots
- `out/fig_zipf_panels.png` - Four-panel Zipf analysis
- `out/fig_heaps.png` - Heaps law with confidence intervals
- `out/fig_yearly_comparison.png` - Yearly word frequency comparison
- `out/fig_growth.png` - Year-over-year growth chart

### Word Clouds
- `out/cloud_overall.png` - Overall corpus word cloud
- `out/cloud_YYYY.png` - Individual yearly word clouds (2017-2025)
- `out/cloud_complete.png` - Complete dataset word cloud

📁 **Backup Location**: All word clouds are backed up to [`.github/assets/wordclouds/`](../.github/assets/wordclouds/) for preservation and reference.

### Report
- `out/report.md` - Complete linguistic analysis report

## 🔬 Validation

### Self-Check Features
- **N-gram Detection**: Validates ngram_lengths_detected covers 1–MAX_N
- **Frequency Thresholds**: Automatically adjusts if no n-grams detected
- **Reproducibility**: SEED=42 ensures consistent results
- **Error Handling**: Graceful fallbacks for missing dependencies/data

### Quality Assurance
- ✅ Semantic filtering for single-character words
- ✅ N-gram semantic coherence validation
- ✅ Zipf's Law compliance verification
- ✅ Multi-dimensional statistical cross-validation
- ✅ English-Chinese mixed content intelligent processing
- ✅ Technical terminology preservation and classification

## 🎯 Design Philosophy

This implementation follows the principle of **surgical precision** - making the smallest possible changes to achieve robust linguistic analysis while maintaining:

1. **Modularity**: Clean separation of concerns in pipeline/
2. **Configurability**: All knobs accessible in main.py
3. **Scientific Rigor**: Validated metrics and reproducible results
4. **Aesthetic Excellence**: Beautiful visualizations worthy of publication
5. **Emotional Resonance**: Cheer-up tone to celebrate linguistic journey

## 📚 Dependencies

Core requirements (see requirements.txt):
- pandas, numpy, matplotlib, scikit-learn
- jieba (fallback), pkuseg (preferred)
- wordcloud, pillow (visualization)
- scipy, seaborn (scientific analysis)

## 🎉 Next Steps

1. Install dependencies: `pip install -r requirements.txt`
2. Run full analysis: `python main.py`
3. Explore results in `out/report.md`
4. Adjust parameters in main.py for different analyses
5. Celebrate your linguistic journey! 🌟