# 交互式词云可视化

这个交互式词云工具基于微信公众号文章的词频分析，提供了丰富的交互功能和数据可视化。

## 功能特性

### 🎨 交互式词云
- **年份筛选**: 支持查看全部年份或特定年份的词云
- **动态调整**: 可调整显示的词汇数量 (50-300个)
- **多种配色**: 5种专业配色方案 (自然、科技、温暖、冷色、单色)
- **悬停提示**: 鼠标悬停显示词汇频次
- **点击详情**: 点击词汇查看跨年份频次对比

### 📊 统计信息
- 实时显示总词数、独特词汇数、当前年份、文档数等统计信息
- 词汇在不同年份的频次对比分析

### 📱 响应式设计
- 支持桌面和移动设备
- 自适应布局，优化不同屏幕尺寸的显示效果

## 使用方法

### 1. 生成词频数据
首先需要运行数据提取脚本生成词云所需的数据：

```bash
cd analysis
python extract_wordcloud_data.py
```

这会在 `analysis/out/wordcloud_data.json` 生成词频数据文件。

### 2. 打开交互式词云
直接在浏览器中打开 `analysis/web/index.html` 文件即可使用。

或者使用本地服务器（推荐）：

```bash
# 在 analysis/web 目录下启动简单的HTTP服务器
cd analysis/web
python -m http.server 8000

# 然后在浏览器中访问 http://localhost:8000
```

## 文件结构

```
analysis/
├── web/
│   ├── index.html          # 交互式词云主页面
│   └── README.md           # 本说明文件
├── out/
│   └── wordcloud_data.json # 词频数据文件 (自动生成)
└── extract_wordcloud_data.py # 数据提取脚本
```

## 数据格式

`wordcloud_data.json` 包含以下结构：

```json
{
  "overall": {
    "words": [{"text": "词汇", "size": 频次}, ...],
    "total_words": 总词数,
    "unique_words": 独特词汇数
  },
  "by_year": {
    "2017": {
      "words": [...],
      "total_words": ...,
      "unique_words": ...,
      "documents": 文档数
    },
    ...
  },
  "years": ["2017", "2018", ...],
  "metadata": {
    "total_documents": 总文档数,
    ...
  }
}
```

## 技术实现

- **前端**: HTML5 + CSS3 + 原生JavaScript
- **词云库**: wordcloud2.js
- **数据处理**: Python (jieba分词 + 统计分析)
- **字体支持**: 支持中文显示
- **兼容性**: 现代浏览器 (Chrome, Firefox, Safari, Edge)

## 自定义配置

可以通过修改 `extract_wordcloud_data.py` 中的参数来自定义：

- `max_words_per_year`: 每年最大词汇数 (默认200)
- `max_words_overall`: 总体最大词汇数 (默认500)
- 中文停用词、英文停用词路径等

## 注意事项

1. 首次运行需要安装Python依赖: `pip install -r requirements.txt`
2. 数据提取可能需要几分钟时间，请耐心等待
3. 建议使用本地HTTP服务器访问，避免跨域问题
4. 如需部署到GitHub Pages，可将web目录内容复制到仓库根目录

## 故障排除

### 词云不显示
- 检查 `wordcloud_data.json` 文件是否存在且格式正确
- 确保使用HTTP服务器访问，而非直接打开HTML文件
- 检查浏览器控制台是否有错误信息

### 数据提取失败
- 确认Python环境和依赖包安装正确
- 检查语料库路径是否正确
- 查看错误信息并根据提示解决

### 性能问题
- 减少显示的词汇数量
- 关闭不必要的浏览器标签页
- 使用现代浏览器以获得更好性能