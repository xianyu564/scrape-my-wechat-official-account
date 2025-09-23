# 风格微调训练模块 (Style Fine-tuning Module)

本模块为微信公众号内容提供个人写作风格微调功能，基于 QLoRA/LoRA 技术实现。

## 🎯 设计特点

- **零入侵**: 不修改任何现有代码，完全独立的模块
- **隐私保护**: 所有数据和模型权重仅存储在本地
- **学术友好**: 提供完整的评测方法，可直接用于论文
- **即插即用**: LoRA 适配器可快速加载/卸载

## 📁 模块结构

```
prep/
├── style_sft_builder.py    # 数据预处理脚本

train/
├── lora_sft.yaml          # 训练配置文件

eval/
├── style_eval.ipynb       # 评测工具

docs/
├── LLM_FINE_TUNE_README.md # 详细使用文档

data/                      # 生成的训练数据 (*.jsonl)
outputs/                   # 训练输出 (LoRA 权重)
```

## 🚀 快速开始

### 1. 环境准备

```bash
# 安装依赖
pip install transformers peft trl bitsandbytes
pip install torch datasets accelerate
pip install matplotlib seaborn jieba notebook
```

### 2. 数据生成

```bash
# 从微信备份生成训练数据
python prep/style_sft_builder.py --backup_dir Wechat-Backup --output_dir data

# 输出文件:
# data/sft_train.jsonl    - 训练集 (90%)
# data/sft_val.jsonl      - 验证集 (10%)
# data/dataset_stats.json - 数据统计
```

### 3. 模型训练

```bash
# 一键训练 (需要GPU)
python train_lora.py

# 训练完成后模型保存在: outputs/qwen25-7b-sft-lora/
```

### 4. 模型测试

```bash
# 快速测试
python test_lora.py

# 对比基础模型和LoRA模型
python test_lora.py --compare --prompt "用我的口吻写一段关于早晨的感受"

# 交互模式
python test_lora.py --interactive
```

### 5. 效果评测

```bash
# 打开 Jupyter Notebook 进行全面评测
jupyter notebook eval/style_eval.ipynb
```

## 📊 数据处理流程

1. **内容扫描**: 递归扫描 `Wechat-Backup/**/` 下的 `.md` 文件
2. **内容清理**: 去除 Markdown 格式、图片链接等
3. **智能切块**: 按 300-800 tokens 切分文本
4. **模板生成**: 生成改写、续写、总结三种训练模板
5. **去重过滤**: 基于内容哈希的重复检测
6. **格式输出**: 标准 JSONL 格式，兼容 Transformers

## 🎛️ 配置说明

### 训练配置 (train/lora_sft.yaml)

```yaml
# 基础模型
model_name: Qwen/Qwen2.5-7B-Instruct

# LoRA 参数
lora_r: 16                 # LoRA rank
lora_alpha: 32             # LoRA alpha
lora_dropout: 0.05         # Dropout 率

# 训练参数  
max_steps: 1500            # 训练步数
learning_rate: 2.0e-4      # 学习率
per_device_train_batch_size: 1

# 量化配置
load_in_4bit: true         # 4-bit 量化
bnb_4bit_quant_type: nf4   # 量化类型
```

### 数据配置

```bash
# 调整数据分割比例
python prep/style_sft_builder.py --train_ratio 0.8

# 处理特定年份
python prep/style_sft_builder.py --filter_years 2023,2024

# 自定义输出目录
python prep/style_sft_builder.py --output_dir custom_data/
```

## 📈 评测维度

### 1. 困惑度 (Perplexity)
- 在验证集上比较基座模型 vs LoRA 模型
- 越低越好，表示模型对文本的预测能力

### 2. 风格指示器  
- 字数/句长分布对比
- 高频词使用频率
- 停用词占比分析

### 3. A/B 人工评测
- 生成盲评样本
- 评估"更像作者"的程度
- 可用于论文的主观评测

## 🔒 隐私与合规

- **本地存储**: 所有数据仅在本地处理，不上传第三方
- **版权保护**: 遵循原内容的 CC BY-NC-SA 4.0 协议
- **AI 声明**: 工具开发使用了生成式 AI 辅助，均经人工审核

## 🛠️ 故障排除

### 显存不足
```bash
# 使用更小的模型
python prep/style_sft_builder.py --model_size 1.5B

# 减小批次大小
# 编辑 train/lora_sft.yaml: per_device_train_batch_size: 1
```

### 训练不收敛
```bash
# 检查数据质量
python prep/style_sft_builder.py --debug

# 调整学习率
# 编辑 train/lora_sft.yaml: learning_rate: 1.0e-4
```

## 📖 高级用法

### 自定义模板

编辑 `prep/style_sft_builder.py` 中的 `generate_training_templates` 函数：

```python
def generate_training_templates(chunk: str, meta_info: Dict) -> List[Dict]:
    templates = []
    
    # 添加新的模板类型
    templates.append({
        "system": "你是该公众号作者，保持其叙述节奏与转折。",
        "input": f"用我的风格总结这段内容：{chunk}",
        "output": f"总结：{chunk[:100]}...",  # 自定义输出
        "meta": meta_info
    })
    
    return templates
```

### 多模型对比

```python
# 测试不同基础模型
models = [
    "Qwen/Qwen2.5-7B-Instruct",
    "Qwen/Qwen2.5-1.5B-Instruct",
    "meta-llama/Llama-2-7b-chat-hf"
]

for model in models:
    python test_lora.py --base_model {model} --compare
```

## 📄 相关文档

- [详细使用指南](docs/LLM_FINE_TUNE_README.md)
- [评测方法说明](eval/style_eval.ipynb)
- [项目未来规划](docs/FUTURE_VISION.md)

## 🤝 贡献指南

欢迎提交 Issue 和 Pull Request！

1. Fork 项目
2. 创建功能分支: `git checkout -b feature/new-feature`
3. 提交修改: `git commit -m 'Add new feature'`
4. 推送分支: `git push origin feature/new-feature`
5. 提交 Pull Request

---

> **免责声明**: 本工具仅供学术研究和个人学习使用。用户应遵守相关法律法规，不得将生成内容用于虚假信息传播或其他有害用途。