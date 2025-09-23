# Style SFT (Supervised Fine-Tuning) 准备目录

这是风格训练准备目录，包含从 wechat-backup 导出数据到 SFT jsonl 的脚本。

## 文件说明

- `prep_style_sft_builder.py` - 数据清洗+切块+生成 JSONL 的主脚本
- `sample_config.yaml` - 配置文件（输入/输出路径等参数）  
- `data/` - 存放导出的 jsonl 文件（不入库）

## 功能特性

1. **智能数据处理**: 遍历 `wechat-backup/**/` 下的 markdown 文件和 meta.json
2. **高质量文本提取**: 从 markdown 中提取正文段落，过滤标题、短句和图片占位符
3. **智能文本切块**: 将正文按 300–800 token 切分，保持语义完整性
4. **多样化训练模板**: 支持改写、续写、总结展开等多种模板类型
5. **高级去重机制**: 支持哈希去重和语义相似度去重
6. **质量过滤**: 自动过滤低质量文本块
7. **灵活配置**: 支持多种参数调整和模板开关

## 运行说明

```bash
# 使用默认配置运行
python prep_style_sft_builder.py --config sample_config.yaml

# 或者直接指定参数
python prep_style_sft_builder.py --input_dir ../Wechat-Backup --output_dir data
```

## 输出结果

脚本将在 `style_sft/data/` 目录下生成：
- `sft_train.jsonl` - 训练集（90%）
- `sft_val.jsonl` - 验证集（10%）
- `dataset_stats.json` - 数据集统计信息

每条样本格式为：
```json
{
  "system": "你是该公众号作者，保持其叙述节奏与转折。",
  "input": "用我的口吻改写下面这段材料：...",
  "output": "原文内容..."
}
```

## 配置选项

### 基础配置
- `input_dir`: 输入目录路径
- `filter_years`: 年份过滤
- `min_chunk_length/max_chunk_length`: 文本块大小控制

### 模板配置
- `enable_rewrite_template`: 启用改写模板
- `enable_continue_template`: 启用续写模板
- `enable_summarize_template`: 启用总结展开模板
- `enable_qa_template`: 启用问答模板（实验性）

### 质量控制
- `enable_deduplication`: 基础去重
- `enable_advanced_dedup`: 高级语义去重
- `enable_quality_filter`: 质量过滤
- `similarity_threshold`: 相似度阈值

## 依赖要求

确保已安装必要的 Python 库：
```bash
pip install pyyaml
```