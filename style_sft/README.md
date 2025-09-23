# Style SFT (Supervised Fine-Tuning) 准备目录

这是风格训练准备目录，包含从 wechat-backup 导出数据到 SFT jsonl 的脚本。

## 文件说明

- `prep_style_sft_builder.py` - 数据清洗+切块+生成 JSONL 的主脚本
- `sample_config.yaml` - 配置文件（输入/输出路径等参数）  
- `data/` - 存放导出的 jsonl 文件（不入库）

## 功能特性

1. **遍历备份数据**: 扫描 `wechat-backup/**/` 下的 markdown 文件和 meta.json
2. **智能文本处理**: 从 markdown 中提取正文段落，过滤标题、短句和图片占位符
3. **智能切块**: 将正文按 300–800 token 切分成合适的训练块
4. **多样化模板**: 生成改写、续写等多种训练样本类型
5. **标准输出**: 输出符合 SFT 训练格式的 JSONL 文件

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
  "output": "原文内容...",
  "meta": {
    "year": "2023",
    "topic": "日常随想"
  }
}
```

## 依赖要求

确保已安装必要的 Python 库：
```bash
pip install pyyaml
```