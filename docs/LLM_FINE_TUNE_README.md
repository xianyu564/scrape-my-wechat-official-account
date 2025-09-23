# LLM 风格微调使用指南

## 概述

本项目提供了完整的个人写作风格微调解决方案，基于 QLoRA/LoRA 技术，实现在不改动原有代码库的前提下，对大语言模型进行个性化风格适配。

## 🎯 设计目标

- **最小入侵**: 不修改任何现有的抓取器代码，新增功能独立于原系统
- **隐私保护**: 所有训练数据和模型权重仅存储在本地，不上传到第三方
- **学术合规**: 提供完整的复现文档和评测方法，可直接用于论文写作
- **即插即用**: 训练后的 LoRA 适配器可快速加载/卸载，不影响基础模型

## 📁 文件结构

```
prep/style_sft_builder.py        # 数据预处理：从 Markdown + meta.json 生成训练数据
train/lora_sft.yaml              # 训练配置：QLoRA/LoRA 超参数设置
eval/style_eval.ipynb            # 评测工具：困惑度、风格指标、A/B测试
docs/LLM_FINE_TUNE_README.md     # 本文档：使用说明和合规指南
```

## 🚀 快速开始

### 1. 环境准备

```bash
# 安装必要依赖
pip install transformers peft trl bitsandbytes
pip install torch torchvision torchaudio
pip install datasets accelerate
pip install matplotlib seaborn jieba notebook
```

### 2. 数据准备

```bash
# 从微信备份生成训练数据
python prep/style_sft_builder.py --backup_dir Wechat-Backup --output_dir data

# 检查生成的数据
ls data/
# 应该看到: sft_train.jsonl, sft_val.jsonl, dataset_stats.json
```

数据处理流程：
- 扫描 `Wechat-Backup/**/` 下的 `.md` 文件和 `meta.json`
- 清理 Markdown 格式，按 300-800 tokens 切块
- 生成三种训练模板：改写、续写、总结展开
- 输出标准的 JSONL 格式训练数据

### 3. 模型训练

创建简单的训练脚本 `train_lora.py`：

```python
#!/usr/bin/env python3
"""
一键式 LoRA 风格微调脚本
"""
import yaml
import torch
from transformers import (
    AutoModelForCausalLM, 
    AutoTokenizer, 
    BitsAndBytesConfig,
    TrainingArguments
)
from trl import SFTTrainer, SFTConfig
from peft import LoraConfig, get_peft_model
import datasets

def main():
    # 加载配置
    with open("train/lora_sft.yaml", 'r', encoding='utf-8') as f:
        cfg = yaml.safe_load(f)
    
    # 量化配置
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=cfg["load_in_4bit"],
        bnb_4bit_quant_type=cfg["bnb_4bit_quant_type"],
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
    )
    
    # 加载tokenizer和模型
    tokenizer = AutoTokenizer.from_pretrained(cfg["model_name"], use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    model = AutoModelForCausalLM.from_pretrained(
        cfg["model_name"],
        quantization_config=bnb_config,
        device_map="auto",
        torch_dtype=torch.float16
    )
    
    # LoRA配置
    peft_config = LoraConfig(
        r=cfg["lora_r"],
        lora_alpha=cfg["lora_alpha"],
        lora_dropout=cfg["lora_dropout"],
        target_modules=cfg["target_modules"],
        bias="none",
        task_type="CAUSAL_LM"
    )
    
    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()
    
    # 加载数据集
    train_dataset = datasets.load_dataset("json", data_files=cfg["dataset"])["train"]
    eval_dataset = datasets.load_dataset("json", data_files=cfg["eval_dataset"])["train"]
    
    # 格式化函数
    def format_prompts(examples):
        texts = []
        for system, input_text, output in zip(examples["system"], examples["input"], examples["output"]):
            text = f"<|system|>\n{system}\n<|user|>\n{input_text}\n<|assistant|>\n{output}<|end|>"
            texts.append(text)
        return {"text": texts}
    
    train_dataset = train_dataset.map(format_prompts, batched=True)
    eval_dataset = eval_dataset.map(format_prompts, batched=True)
    
    # 训练配置
    training_args = SFTConfig(
        output_dir=cfg["output_dir"],
        per_device_train_batch_size=cfg["per_device_train_batch_size"],
        gradient_accumulation_steps=cfg["gradient_accumulation_steps"],
        learning_rate=cfg["learning_rate"],
        max_steps=cfg["max_steps"],
        logging_steps=cfg["logging_steps"],
        save_steps=cfg["save_steps"],
        evaluation_strategy="steps",
        eval_steps=cfg.get("eval_steps", 200),
        bf16=cfg.get("bf16", True),
        gradient_checkpointing=True,
        optim="paged_adamw_32bit",
        lr_scheduler_type=cfg.get("lr_scheduler_type", "cosine"),
        warmup_steps=cfg.get("warmup_steps", 100),
        max_seq_length=cfg.get("max_seq_length", 1024),
        packing=False,
        dataset_text_field="text"
    )
    
    # 创建训练器
    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
    )
    
    # 开始训练
    print("开始训练...")
    trainer.train()
    
    # 保存模型
    trainer.save_model()
    print(f"模型已保存到: {cfg['output_dir']}")

if __name__ == "__main__":
    main()
```

运行训练：

```bash
python train_lora.py
```

预期训练时间：
- 7B 模型 + QLoRA: ~2-4小时 (24GB GPU)
- 1.5B 模型: ~30-60分钟 (16GB GPU)

### 4. 模型评测

打开 Jupyter Notebook 进行评测：

```bash
jupyter notebook eval/style_eval.ipynb
```

评测包含三个维度：
1. **困惑度 (PPL)**: 在验证集上比较基座模型 vs LoRA 模型
2. **风格指示器**: 字数分布、词频、停用词比例等统计特征对比
3. **A/B 人工评测**: 生成盲评样本，评估"更像作者"的程度

### 5. 模型推理

创建推理脚本进行测试：

```python
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel
import torch

# 配置
base_model_name = "Qwen/Qwen2.5-7B-Instruct"
lora_path = "outputs/qwen25-7b-sft-lora"

# 加载模型
bnb_config = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4")
tokenizer = AutoTokenizer.from_pretrained(base_model_name, use_fast=True)
model = AutoModelForCausalLM.from_pretrained(
    base_model_name, 
    quantization_config=bnb_config, 
    device_map="auto"
)
model = PeftModel.from_pretrained(model, lora_path)

# 生成文本
prompt = "用我的口吻写一段关于早晨的感受："
inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
outputs = model.generate(
    **inputs, 
    max_new_tokens=200, 
    temperature=0.7,
    do_sample=True,
    pad_token_id=tokenizer.eos_token_id
)
result = tokenizer.decode(outputs[0], skip_special_tokens=True)
print(result)
```

## 📊 评测标准

### 定量指标
- **困惑度降低**: LoRA 模型相对基座模型的 PPL 改进
- **风格一致性**: 字数、句长、词频分布与原文的相似度
- **词汇多样性**: 生成文本的词汇丰富程度

### 定性评估
- **A/B 盲评**: 随机样本对比，评估"更像作者"的胜率
- **语义连贯性**: 生成内容的逻辑性和流畅度
- **风格特征**: 是否保持了原作者的语言习惯和表达方式

## 🔒 隐私与合规

### 数据保护
- **本地存储**: 所有原始语料、训练数据、模型权重仅存储在本地
- **不上传策略**: 严禁将个人语料或适配器权重上传到公共平台
- **访问控制**: 训练环境应设置适当的访问权限控制

### 使用合规
- **版权声明**: 原始文章内容遵循 CC BY-NC-SA 4.0 协议
- **代码开源**: 工具代码遵循 MIT 协议
- **学术使用**: 本项目适用于个人学习和学术研究，商业使用需额外授权

### AI 工具披露
- 本项目在开发过程中使用了生成式 AI 工具进行代码脚手架搭建
- 所有生成内容均经过人工审核和修改
- AI 工具仅作为辅助，不享有署名权

## 🔧 故障排除

### 常见问题

**Q: 显存不足怎么办？**
A: 
1. 使用更小的基础模型（如 Qwen2.5-1.5B）
2. 减小 batch_size 和 max_seq_length
3. 启用 gradient_checkpointing
4. 使用 8-bit 量化替代 4-bit

**Q: 训练损失不下降？**
A:
1. 检查学习率是否过大/过小
2. 确认数据格式正确
3. 增加 warmup_steps
4. 检查 LoRA rank 设置

**Q: 生成效果不理想？**
A:
1. 增加训练步数或数据量
2. 调整 LoRA 参数 (r, alpha)
3. 检查训练数据质量
4. 尝试不同的采样策略

### 性能优化

- **训练加速**: 使用 `flash-attention-2` 加速注意力计算
- **内存优化**: 启用 `gradient_checkpointing` 和 `dataloader_pin_memory`
- **并行训练**: 多卡环境下使用 `accelerate` 进行分布式训练

## 📖 学术应用

本项目提供的评测方法和结果可直接用于以下学术场景：

### 论文章节
- **方法论**: LoRA 微调技术和数据处理流程
- **实验设计**: 困惑度测试、风格指标分析、A/B 评测
- **结果分析**: 定量指标对比和定性效果评估

### 可复现性
- 完整的配置文件和训练脚本
- 详细的环境依赖和版本说明  
- 标准化的评测流程和指标计算

### 合规要求
- 数据使用和隐私保护说明
- AI 工具使用声明
- 开源协议和版权信息

## 🛣️ 未来扩展

### 技术改进
- 支持更多基础模型（LLaMA, ChatGLM, Baichuan）
- 集成更先进的微调技术（QLoRA, AdaLoRA）
- 添加多模态能力（图文混合生成）

### 功能增强
- Web UI 界面简化使用流程
- 自动化评测管道
- 多作者风格对比分析

## 📞 技术支持

如有技术问题或改进建议，请通过以下方式联系：
- GitHub Issues: 项目问题追踪
- 技术文档: `docs/` 目录下的详细说明
- 社区讨论: 加入相关技术交流群

---

**免责声明**: 本工具仅供学术研究和个人学习使用。用户应遵守相关法律法规，不得将生成内容用于虚假信息传播或其他有害用途。