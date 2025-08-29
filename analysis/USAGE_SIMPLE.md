# 中文语料分析系统 - 简化版本

## 🎯 设计理念

根据用户反馈，本系统采用**分两步走**的设计：

1. **第一步：理论分析** - 纯粹处理文本语料，不产生可视化输出
2. **第二步：可视化呈现** - 基于第一步结果生成词云、图表等美观的交付成果

这样的分离设计便于用户调整参数，特别是可视化参数，而无需重复耗时的文本分析过程。

## 🚀 使用方法

### 直接运行主程序

```bash
cd analysis
python main.py
```

### 参数配置

所有参数都在 `main.py` 的 `main()` 函数中配置，采用注释的方式列出所有可选项：

```python
def main():
    # =================================================================
    # 🔧 配置参数 - 根据需要修改下面的参数
    # =================================================================
    
    # 必需参数
    ROOT_DIR = "Wechat-Backup/文不加点的张衔瑜"  # 语料根目录
    OUTPUT_DIR = "analysis/out"                    # 输出目录
    
    # 运行模式选择
    RUN_ANALYSIS = True       # 是否运行第一步（理论分析）
    RUN_VISUALIZATION = True  # 是否运行第二步（可视化呈现）
    
    # 第一步：分析参数
    ANALYSIS_PARAMS = {
        'start_date': None,           # "2020-01-01" 或 None
        'end_date': None,             # "2023-12-31" 或 None  
        'years': None,                # ["2021", "2022", "2023"] 或 None
        'min_df': 5,                  # TF-IDF最小文档频率
        'max_df': 0.85,               # TF-IDF最大文档频率
        'ngram_max': 2,               # 最大n-gram长度
        # ... 更多选项见代码注释
    }
    
    # 第二步：可视化参数
    VISUALIZATION_PARAMS = {
        'make_wordcloud': True,              # 是否生成词云
        'wordcloud_top_n': 200,              # 整体词云词汇数量
        'font_path': None,                   # 中文字体文件路径
        # ... 更多选项见代码注释
    }
```

## 📊 生成结果

### 第一步输出
- `freq_overall.csv` - 整体词频统计
- `freq_by_year.csv` - 年度词频统计  
- `tfidf_topk_by_year.csv` - TF-IDF关键词（如果成功）
- `analysis_results.pkl` - 分析结果缓存（供第二步使用）

### 第二步输出
- `zipf_overall.png` - Zipf定律分析图
- `wordcloud_overall.png` - 整体词云
- `wordcloud_YYYY.png` - 年度词云
- `REPORT.md` - 综合分析报告

## 🎨 灵活使用

1. **完整分析**：设置 `RUN_ANALYSIS = True, RUN_VISUALIZATION = True`
2. **只做分析**：设置 `RUN_ANALYSIS = True, RUN_VISUALIZATION = False`
3. **只做可视化**：设置 `RUN_ANALYSIS = False, RUN_VISUALIZATION = True`（需要先运行过第一步）

第三种模式特别适合：
- 调试可视化参数
- 尝试不同的词云样式
- 生成不同格式的报告

## 💡 实际使用示例

```python
# 示例1：只分析2023年的数据
ANALYSIS_PARAMS = {
    'years': ["2023"],
    'min_df': 3,
    'ngram_max': 2,
}

# 示例2：生成高清词云  
VISUALIZATION_PARAMS = {
    'wordcloud_top_n': 300,
    'font_path': "/path/to/chinese-font.ttf",
}

# 示例3：只重新生成可视化
RUN_ANALYSIS = False
RUN_VISUALIZATION = True
```

这种设计让用户可以轻松调整参数并快速看到效果，而不需要使用复杂的命令行参数。