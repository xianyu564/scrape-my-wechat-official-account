# 词云生成指南 / Word Cloud Generation Guide

## 可用的词云文件 / Available Word Cloud Files

### 年度词云 / Yearly Word Clouds
- `out/cloud_2017.png` - 2017年词云 (原始高质量版本)
- `out/cloud_2018.png` - 2018年词云 (原始高质量版本)  
- `out/cloud_2019.png` - 2019年词云 (原始高质量版本)
- `out/cloud_2020.png` - 2020年词云 (新生成，支持中文)
- `out/cloud_2021.png` - 2021年词云 (新生成，支持中文)
- `out/cloud_2022.png` - 2022年词云 (新生成，支持中文)
- `out/cloud_2023.png` - 2023年词云 (新生成，支持中文)
- `out/cloud_2024.png` - 2024年词云 (新生成，支持中文)
- `out/cloud_2025.png` - 2025年词云 (新生成，支持中文)

### 整体词云 / Complete Dataset Word Clouds
- `out/cloud_overall.png` - 原始完整数据词云 (2017-2019)
- `out/cloud_complete.png` - 完整数据词云 (2020-2025)

## 自定义词云生成 / Custom Word Cloud Generation

使用 `generate_wordclouds.py` 工具生成自定义词云:

### 基本用法 / Basic Usage

```bash
# 生成所有年份的词云
python generate_wordclouds.py

# 生成特定年份的词云
python generate_wordclouds.py --years 2020,2021,2022

# 生成特定时间段的词云 (如您要求的 2020.08-2024.12)
python generate_wordclouds.py --start-date 2020-08-01 --end-date 2024-12-31

# 自定义输出目录
python generate_wordclouds.py --output custom_clouds --years 2023,2024
```

### 高级参数 / Advanced Parameters

```bash
# 自定义词数和颜色方案
python generate_wordclouds.py --max-words 300 --color-scheme science

# 指定中文字体
python generate_wordclouds.py --font-path /path/to/chinese/font.ttf

# 完整示例：生成您指定的时间段
python generate_wordclouds.py \
  --start-date 2020-08-01 \
  --end-date 2024-12-31 \
  --output period_2020_2024 \
  --max-words 250 \
  --color-scheme nature
```

### 可用的颜色方案 / Available Color Schemes
- `nature` (默认) - 自然色调
- `science` - 科技蓝调
- `calm` - 平静色调
- `muted` - 柔和色调
- `solar` - 暖阳色调

## 系统特性 / System Features

✅ **自动中文字体检测** - 系统会自动找到并使用可用的中文字体  
✅ **智能文件保护** - 不会覆盖已存在的原始优质词云文件  
✅ **时间段筛选** - 支持精确的日期范围筛选  
✅ **多种输出格式** - 支持年度词云和完整数据词云  
✅ **自定义参数** - 可调整词数、颜色方案、字体等参数

## 故障排除 / Troubleshooting

**如果中文显示为方框:**
1. 系统已安装中文字体 (WenQuanYi, Noto CJK)
2. 脚本会自动检测和使用中文字体
3. 如需指定特定字体，使用 `--font-path` 参数

**如果需要重新生成词云:**
1. 删除对应的 `.png` 文件
2. 重新运行生成命令
3. 原始的 2017-2019 词云会被保护不被覆盖

**示例输出文件:**
- 生成的词云文件大小约 1.5-2.8MB
- 支持中文字符完整显示
- 高质量 PNG 格式输出