# Chinese Linguistic Analysis

中文语料的整体与逐年词频分析、TF-IDF 关键词与高颜值词云分析系统。

## 功能特性

- **语料装载**: 自动扫描 Wechat-Backup 目录结构，提取所有文章
- **中文分词**: 基于 jieba 的精确分词，支持自定义词典和停用词
- **统计分析**: 词频统计、TF-IDF 关键词提取、Zipf 定律分析
- **词云可视化**: 高颜值中文词云，支持自定义形状和配色
- **命令行工具**: 一键生成分析报告
- **Jupyter 笔记本**: 交互式分析和可视化

## 安装依赖

```bash
pip install -r requirements.txt
```

## 使用方法

### 命令行使用

```bash
# 基础分析
python -m analysis.src.cli --root Wechat-Backup/文不加点的张衔瑜

# 带时间过滤
python -m analysis.src.cli --root Wechat-Backup/文不加点的张衔瑜 --start 2020-01-01 --end 2023-12-31

# 指定年份
python -m analysis.src.cli --root Wechat-Backup/文不加点的张衔瑜 --years 2021,2022,2023

# 完整配置
python -m analysis.src.cli \
  --root Wechat-Backup/文不加点的张衔瑜 \
  --min-df 5 --max-df 0.85 --ngram-max 2 \
  --font-path /path/to/font.ttf \
  --topk 50 \
  --make-wordcloud 1 \
  --report 1
```

### 输出文件

运行后会在 `analysis/out/` 目录生成：

- `freq_overall.csv` - 整体词频统计
- `freq_by_year.csv` - 逐年词频统计  
- `tfidf_topk_by_year.csv` - 年度 TF-IDF 关键词
- `zipf_overall.png` - Zipf 定律分析图
- `wordcloud_overall.png` - 整体词云
- `wordcloud_<YYYY>.png` - 年度词云
- `REPORT.md` - 分析报告

## 目录结构

```
analysis/
├── README_analysis.md          # 本文档
├── requirements.txt           # 依赖包
├── src/                      # 源代码
│   ├── __init__.py
│   ├── io_utils.py           # 语料装载
│   ├── tokenize_zh.py        # 中文分词
│   ├── freq_stats.py         # 统计分析
│   ├── wordcloud_viz.py      # 词云可视化
│   └── cli.py                # 命令行工具
├── notebooks/                # Jupyter 笔记本
│   └── linguistic_report.ipynb
├── out/                      # 输出文件
├── assets/                   # 资源文件
│   ├── mask.png             # 词云蒙版
│   └── stopwords_zh.txt     # 中文停用词
└── .cache/                   # 缓存文件
```

## 技术架构

- **数据装载**: 自动识别 WeChat 备份目录结构
- **文本预处理**: jieba 分词 + 正则表达式清洗
- **统计分析**: scikit-learn TF-IDF + pandas 数据处理
- **可视化**: matplotlib + wordcloud + 自定义配色
- **缓存机制**: 分词结果缓存，提升重复分析性能

## 质量保证

- 异常处理：文件读取错误不中断流程
- 增量处理：已存在输出文件时跳过
- 进度显示：大文件处理时显示 tqdm 进度条
- 单元测试：核心模块的可靠性测试