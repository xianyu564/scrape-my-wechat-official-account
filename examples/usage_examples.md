# 风格微调使用示例

## 完整工作流程示例

### 步骤 1: 数据准备

```bash
# 生成训练数据
python prep/style_sft_builder.py \
    --backup_dir Wechat-Backup \
    --output_dir data \
    --train_ratio 0.9 \
    --seed 42

# 查看数据统计
cat data/dataset_stats.json
```

输出示例:
```json
{
  "total_articles": 317,
  "total_samples": 3619,
  "train_samples": 3257,
  "val_samples": 362,
  "train_ratio": 0.9,
  "years": ["2017", "2018", "2019", "2020", "2021", "2022", "2023", "2024", "2025"],
  "sample_templates": ["改写", "续写", "总结展开"]
}
```

### 步骤 2: 训练配置

编辑 `train/lora_sft.yaml`:

```yaml
# 基础模型配置 (可根据显存调整)
model_name: Qwen/Qwen2.5-7B-Instruct  # 或 Qwen/Qwen2.5-1.5B-Instruct

# LoRA 配置
lora_r: 16
lora_alpha: 32
lora_dropout: 0.05

# 训练配置
max_steps: 1500
learning_rate: 2.0e-4
per_device_train_batch_size: 1  # 显存不足可设为 1

# 4-bit 量化 (节省显存)
load_in_4bit: true
bnb_4bit_quant_type: nf4
```

### 步骤 3: 开始训练

```bash
# 一键训练
python train_lora.py

# 预期输出:
# 开始 LoRA 风格微调训练...
# 使用模型: Qwen/Qwen2.5-7B-Instruct
# 训练数据: data/sft_train.jsonl
# 验证数据: data/sft_val.jsonl
# 正在加载tokenizer和模型...
# LoRA 配置完成:
# trainable params: 41,943,040 || all params: 7,658,842,112 || trainable%: 0.55
# 训练样本数: 3257
# 验证样本数: 362
# 开始训练...
# 预计训练步数: 1500
# 保存路径: outputs/qwen25-7b-sft-lora
```

### 步骤 4: 模型测试

```bash
# 快速测试
python test_lora.py

# 输出示例:
# 正在加载基础模型: Qwen/Qwen2.5-7B-Instruct
# 正在加载LoRA适配器: outputs/qwen25-7b-sft-lora
# 模型加载完成!
# 运行默认测试用例...
# 
# === 测试 1/5 ===
# 提示: 用我的口吻写一段关于早晨的感受
# 生成: 早晨的光线总是带着一种不确定性...

# 对比基础模型和微调模型
python test_lora.py --compare --prompt "用我的口吻谈谈对读书的看法"

# 交互模式
python test_lora.py --interactive
```

### 步骤 5: 评测分析

```bash
# 启动 Jupyter 进行完整评测
jupyter notebook eval/style_eval.ipynb
```

评测内容包括:
1. **困惑度对比**: 基础模型 vs LoRA 模型
2. **风格指标**: 字数分布、词频、停用词比例
3. **A/B 测试**: 生成盲评样本用于人工评测

## 高级配置示例

### 小显存配置 (适用于 RTX 3060 12GB)

```yaml
# train/lora_sft.yaml
model_name: Qwen/Qwen2.5-1.5B-Instruct  # 使用更小的模型
per_device_train_batch_size: 1
gradient_accumulation_steps: 16          # 增加梯度累积
max_seq_length: 512                      # 减小序列长度
load_in_4bit: true
gradient_checkpointing: true
```

### 高质量配置 (适用于 RTX 4090 24GB+)

```yaml
# train/lora_sft.yaml  
model_name: Qwen/Qwen2.5-7B-Instruct
per_device_train_batch_size: 2
gradient_accumulation_steps: 4
max_seq_length: 1024
max_steps: 2000                          # 更多训练步数
lora_r: 32                               # 更高的 LoRA rank
learning_rate: 1.5e-4                    # 更保守的学习率
```

## 常见问题解决

### Q1: 显存不足 (CUDA out of memory)

**解决方案**:
```bash
# 1. 使用更小的模型
# 编辑 train/lora_sft.yaml: model_name: Qwen/Qwen2.5-1.5B-Instruct

# 2. 减小批次大小
# 编辑 train/lora_sft.yaml: per_device_train_batch_size: 1

# 3. 减小序列长度
# 编辑 train/lora_sft.yaml: max_seq_length: 512

# 4. 启用梯度检查点
# 编辑 train/lora_sft.yaml: gradient_checkpointing: true
```

### Q2: 训练损失不下降

**解决方案**:
```bash
# 1. 检查数据质量
python prep/style_sft_builder.py --backup_dir Wechat-Backup --output_dir data --train_ratio 0.95

# 2. 调整学习率
# 编辑 train/lora_sft.yaml: learning_rate: 1.0e-4

# 3. 增加 warmup
# 编辑 train/lora_sft.yaml: warmup_steps: 200

# 4. 检查数据格式
head -3 data/sft_train.jsonl
```

### Q3: 生成效果不理想

**解决方案**:
```bash
# 1. 增加训练步数
# 编辑 train/lora_sft.yaml: max_steps: 2500

# 2. 调整 LoRA 参数
# 编辑 train/lora_sft.yaml: lora_r: 32, lora_alpha: 64

# 3. 检查推理参数
python test_lora.py --prompt "测试提示" --temperature 0.8 --max_tokens 300
```

## 自定义扩展

### 添加新的训练模板

编辑 `prep/style_sft_builder.py`:

```python
def generate_training_templates(chunk: str, meta_info: Dict) -> List[Dict]:
    templates = []
    
    # 原有模板...
    
    # 新增：仿写模板
    if len(chunk) > 300:
        templates.append({
            "system": "你是该公众号作者，保持其叙述节奏与转折。",
            "input": f"仿照下面的写作风格，写一段类似的文字：{chunk[:150]}...",
            "output": chunk,
            "meta": meta_info
        })
    
    return templates
```

### 添加年份过滤

编辑 `prep/style_sft_builder.py`:

```python
def load_wechat_backup_data(backup_dir: Path, filter_years=None) -> List[Tuple[str, Dict]]:
    articles = []
    
    for md_file in backup_dir.rglob("*.md"):
        # ... 现有代码 ...
        
        # 年份过滤
        if filter_years and meta_info.get('year') not in filter_years:
            continue
            
        # ... 继续处理 ...
```

使用:
```bash
python prep/style_sft_builder.py --filter_years 2023,2024,2025
```

---

更多详细信息请参考 [完整文档](docs/LLM_FINE_TUNE_README.md)