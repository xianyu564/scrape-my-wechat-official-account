# 词云生成指南 / Word Cloud Generation Guide

## 当前输出文件 / Current Output Files

### 词云文件 / Word Cloud Files
- `cloud_overall.png` - 所有年份完整词云 / Complete word cloud for all years (2017-2025)
- `cloud_2017.png` - 2017年词云 / 2017 word cloud
- `cloud_2018.png` - 2018年词云 / 2018 word cloud  
- `cloud_2019.png` - 2019年词云 / 2019 word cloud
- `cloud_2020.png` - 2020年词云 / 2020 word cloud
- `cloud_2021.png` - 2021年词云 / 2021 word cloud
- `cloud_2022.png` - 2022年词云 / 2022 word cloud
- `cloud_2023.png` - 2023年词云 / 2023 word cloud
- `cloud_2024.png` - 2024年词云 / 2024 word cloud
- `cloud_2025.png` - 2025年词云 / 2025 word cloud

### 其他分析文件 / Other Analysis Files
- `summary.json` - 分析结果摘要 / Analysis summary
- `analysis_results.json` - 完整分析结果 / Complete analysis results
- `analysis_results.xlsx` - Excel格式分析结果 / Excel format analysis results
- `report.md` - Markdown分析报告 / Markdown analysis report
- `comprehensive_report.txt` - 综合文本报告 / Comprehensive text report

## 自定义词云生成 / Custom Word Cloud Generation

使用 `generate_wordclouds.py` 工具可以生成自定义词云：

Use the `generate_wordclouds.py` tool to generate custom word clouds:

### 基本用法 / Basic Usage

```bash
# 生成所有年份的词云 / Generate word clouds for all years
python generate_wordclouds.py

# 生成特定年份的词云 / Generate word clouds for specific years
python generate_wordclouds.py --years 2020,2021,2022,2023,2024,2025

# 生成指定时间段的词云 / Generate word clouds for date range
python generate_wordclouds.py --start-date 2020-01-01 --end-date 2022-12-31
```

### 高级选项 / Advanced Options

```bash
# 自定义输出目录和参数 / Custom output and parameters
python generate_wordclouds.py \
  --output custom_clouds \
  --max-words 300 \
  --color-scheme science

# 使用自定义字体 / Use custom font
python generate_wordclouds.py \
  --font-path /path/to/chinese/font.ttf \
  --color-scheme nature
```

### 参数说明 / Parameter Description

- `--corpus` : 语料库根目录 / Corpus root directory (默认: ../Wechat-Backup/文不加点的张衔瑜)
- `--output` : 输出目录 / Output directory (默认: wordcloud_output)
- `--years` : 年份列表(逗号分隔) / Year list (comma-separated)
- `--start-date` : 开始日期 (YYYY-MM-DD) / Start date
- `--end-date` : 结束日期 (YYYY-MM-DD) / End date
- `--max-words` : 最大词数 / Maximum words (默认: 200)
- `--font-path` : 中文字体路径 / Chinese font path
- `--color-scheme` : 颜色方案 / Color scheme (nature, science, calm, muted, solar)

### 颜色方案 / Color Schemes

1. **nature** (默认) - 自然色调 / Natural tones
2. **science** - 科学期刊风格 / Scientific journal style
3. **calm** - 平静色调 / Calm tones
4. **muted** - 柔和色调 / Muted tones
5. **solar** - 暖色调 / Solar/warm tones

## 使用示例 / Usage Examples

### 1. 生成近年数据词云 / Generate Recent Years Word Cloud
```bash
python generate_wordclouds.py --years 2020,2021,2022,2023,2024,2025 --max-words 150
```

### 2. 生成疫情期间词云 / Generate Pandemic Period Word Cloud
```bash
python generate_wordclouds.py --start-date 2020-01-01 --end-date 2022-12-31 --output pandemic_period
```

### 3. 生成高质量科学风格词云 / Generate High-Quality Scientific Style Word Cloud
```bash
python generate_wordclouds.py --max-words 300 --color-scheme science --output scientific_clouds
```

## 注意事项 / Notes

1. **字体支持** / Font Support: 系统会自动检测中文字体，如显示不正常请指定字体路径
2. **内存使用** / Memory Usage: 大数据集可能需要较多内存，建议分年份生成
3. **输出质量** / Output Quality: 词云以300 DPI高分辨率保存，适合打印和展示

## 故障排除 / Troubleshooting

### 中文字符显示问题 / Chinese Character Display Issues
如果词云中中文字符显示为方块，请：
1. 安装中文字体包：`sudo apt install fonts-noto-cjk`
2. 或者指定字体路径：`--font-path /path/to/font.ttf`

### 内存不足 / Out of Memory
如果处理大数据集时内存不足，请：
1. 分年份生成：`--years 2020,2021`
2. 减少词数：`--max-words 100`
3. 分时间段处理：`--start-date 2020-01-01 --end-date 2020-12-31`