# 交互式中文词云分析工具

一个功能强大的中文文本分析和词云可视化工具，支持多种数据格式、灵活的分词方式和交互式探索。

## 功能特性

### 🔄 数据支持
- **多格式支持**: JSON、CSV、Parquet 文件
- **编码自动检测**: UTF-8、UTF-8-BOM、GBK、GB2312 等
- **递归扫描**: 自动发现子目录中的数据文件
- **智能列映射**: 自动识别中英文列名并映射到标准字段

### 🔤 中文分词
- **Jieba 分词**: 精确模式、搜索引擎模式
- **N-gram 分析**: 1-6 可调，支持中文字符级和英文词级
- **繁简转换**: 可选的繁体转简体合并
- **自定义词典**: 停用词、白名单、同义词映射

### ☁️ 交互式词云
- **ECharts 渲染**: 高质量、可交互的词云可视化
- **点击交互**: 点击词汇查看详细上下文信息
- **KWIC 索引**: 关键词在上下文中的位置显示
- **多种主题**: 冷色调、灰阶、蓝紫高亮

### 🎨 美观设计
- **现代化 UI**: 简洁、科技感的界面设计
- **响应式布局**: 左侧控制面板，右侧结果展示
- **中文字体支持**: 优先使用 Noto Sans SC / Source Han Sans
- **优雅配色**: 基于 #FBFBFD 背景的冷色调设计

## 安装与运行

### 1. 安装依赖

```bash
pip install -r analysis/requirements.txt
```

### 2. 运行应用

```bash
streamlit run analysis/app_wordcloud.py
```

应用将在浏览器中打开，默认地址: http://localhost:8501

## 数据准备

### 支持的文件格式

将数据文件放置在 `analysis/data/` 目录下（可在应用中自定义路径）：

- **JSON 格式**: `*.json`
- **CSV 格式**: `*.csv` 
- **Parquet 格式**: `*.parquet`

### 列名映射规则

应用自动识别以下列名（不区分大小写）：

| 标准字段 | 支持的列名 |
|---------|-----------|
| title | title, 标题, 题目 |
| content | content, text, body, article, 内容, 正文, 文本, 文章 |
| url | url, link, href, 链接, 网址 |
| date | date, published_at, publish_date, created_at, time, 日期, 时间, 发布时间, 创建时间 |

### 数据格式示例

**JSON 格式**:
```json
[
  {
    "title": "文章标题",
    "content": "文章内容...",
    "url": "https://example.com/article1",
    "date": "2024-01-01"
  }
]
```

**CSV 格式**:
```csv
title,content,url,date
文章标题1,这是第一篇文章的内容...,https://example.com/1,2024-01-01
文章标题2,这是第二篇文章的内容...,https://example.com/2,2024-01-02
```

### 单列文本格式

如果数据只有一列纯文本，应用会自动将其映射到 `content` 字段：

```csv
text_content
这是一段长文本内容，会被自动识别为文章内容...
另一段文本内容...
```

## 使用指南

### 1. 加载数据

1. 在侧栏"数据源"部分输入数据目录路径
2. 点击"🔄 加载数据"按钮
3. 查看数据统计信息，确认加载成功

### 2. 配置分词

- **分词模式**:
  - Jieba 精确模式: 适合一般文本分析
  - Jieba 搜索引擎模式: 会产生更多细粒度分词
  - N-gram 模式: 按 N 个字符/词组合，适合发现固定搭配

- **频次设置**:
  - 最小词频: 过滤低频词汇
  - 显示词数: 词云中显示的词汇数量上限

### 3. 自定义资源

#### 停用词文件 (txt)
每行一个停用词，以 # 开头的行为注释：
```
的
是
在
# 这是注释
和
```

#### 白名单文件 (txt)
每行一个要保留的词汇（优先级高于停用词）：
```
人工智能
机器学习
深度学习
```

#### 同义词映射 (csv)
包含 `from` 和 `to` 两列：
```csv
from,to
AI,人工智能
ML,机器学习
大模型,人工智能
AIGC,人工智能
生成式AI,人工智能
```

#### 字体文件 (ttf/otf)
上传中文字体文件以获得更好的显示效果。推荐字体：
- Noto Sans SC
- Source Han Sans
- 思源黑体

### 4. 生成和交互

1. 配置完成后点击"🎨 生成词云"
2. 词云生成后可以：
   - 点击词汇查看详细信息
   - 在右侧查看 KWIC 上下文
   - 浏览词频排行榜
   - 点击原文链接查看完整文章

### 5. 导出结果

- **导出 HTML 词云**: 生成可交互的 HTML 文件
- **导出 CSV 词频表**: 导出词频统计数据

导出文件保存在 `analysis/outputs/` 目录下。

## 故障排除

### 常见问题

1. **数据加载失败**
   - 检查文件路径是否正确
   - 确认文件格式和编码
   - 查看错误详情中的具体提示

2. **中文显示异常**
   - 安装推荐的中文字体
   - 或在侧栏上传自定义字体文件

3. **分词结果不理想**
   - 尝试不同的分词模式
   - 上传自定义停用词和白名单
   - 调整最小词频阈值

4. **词云显示空白**
   - 检查是否有足够的有效词汇
   - 降低最小词频阈值
   - 确认文本内容不为空

### 性能优化

- 对于大型数据集，建议先用小样本测试
- 适当提高最小词频以减少词汇数量
- 使用 Top-K 限制显示词数

## 技术实现

### 核心模块

- `data_loader.py`: 数据加载和格式化
- `text_pipeline.py`: 文本处理和分词
- `app_wordcloud.py`: Streamlit 应用主体

### 依赖库

- **Streamlit**: Web 应用框架
- **Jieba**: 中文分词
- **PyEcharts**: 图表可视化
- **OpenCC**: 繁简转换
- **Pandas**: 数据处理
- **Charset-normalizer**: 编码检测

## 开发和测试

### 运行测试

```bash
# 数据加载测试
python -m unittest analysis.tests.test_loader

# 文本处理测试
python -m unittest analysis.tests.test_text_pipeline

# 运行所有测试
python -m unittest discover analysis.tests
```

### 开发模式

```bash
# 启用开发模式（自动重载）
streamlit run analysis/app_wordcloud.py --server.runOnSave true
```

## 许可证

本项目遵循项目根目录的许可证条款。

## 贡献

欢迎提交 Issue 和 Pull Request 来改进这个工具！