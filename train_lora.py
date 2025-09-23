#!/usr/bin/env python3
"""
一键式 LoRA 风格微调脚本
基于 QLoRA/LoRA 技术，实现个人写作风格微调
"""

import yaml
import torch
import os
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
    print("开始 LoRA 风格微调训练...")
    
    # 加载配置
    config_path = "train/lora_sft.yaml"
    if not os.path.exists(config_path):
        print(f"配置文件 {config_path} 不存在！")
        return
    
    with open(config_path, 'r', encoding='utf-8') as f:
        cfg = yaml.safe_load(f)
    
    print(f"使用模型: {cfg['model_name']}")
    print(f"训练数据: {cfg['dataset']}")
    print(f"验证数据: {cfg['eval_dataset']}")
    
    # 检查数据文件
    if not os.path.exists(cfg['dataset']):
        print(f"训练数据文件 {cfg['dataset']} 不存在！")
        print("请先运行: python prep/style_sft_builder.py")
        return
    
    # 量化配置
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=cfg["load_in_4bit"],
        bnb_4bit_quant_type=cfg["bnb_4bit_quant_type"],
        bnb_4bit_compute_dtype=getattr(torch, cfg.get("bnb_4bit_compute_dtype", "float16")),
        bnb_4bit_use_double_quant=cfg.get("bnb_4bit_use_double_quant", True),
    )
    
    print("正在加载tokenizer和模型...")
    
    # 加载tokenizer和模型
    tokenizer = AutoTokenizer.from_pretrained(cfg["model_name"], use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    model = AutoModelForCausalLM.from_pretrained(
        cfg["model_name"],
        quantization_config=bnb_config,
        device_map="auto",
        torch_dtype=getattr(torch, cfg.get("torch_dtype", "float16"))
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
    print("LoRA 配置完成:")
    model.print_trainable_parameters()
    
    # 加载数据集
    print("正在加载训练数据...")
    try:
        train_dataset = datasets.load_dataset("json", data_files=cfg["dataset"])["train"]
        eval_dataset = datasets.load_dataset("json", data_files=cfg["eval_dataset"])["train"]
    except Exception as e:
        print(f"加载数据集时出错: {e}")
        return
    
    print(f"训练样本数: {len(train_dataset)}")
    print(f"验证样本数: {len(eval_dataset)}")
    
    # 格式化函数
    def format_prompts(examples):
        texts = []
        for system, input_text, output in zip(examples["system"], examples["input"], examples["output"]):
            # 使用 ChatML 格式
            text = f"<|system|>\n{system}\n<|user|>\n{input_text}\n<|assistant|>\n{output}<|end|>"
            texts.append(text)
        return {"text": texts}
    
    # 应用格式化
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
        gradient_checkpointing=cfg.get("gradient_checkpointing", True),
        optim=cfg.get("optim", "paged_adamw_32bit"),
        lr_scheduler_type=cfg.get("lr_scheduler_type", "cosine"),
        warmup_steps=cfg.get("warmup_steps", 100),
        max_seq_length=cfg.get("max_seq_length", 1024),
        packing=cfg.get("packing", False),
        dataset_text_field="text",
        load_best_model_at_end=cfg.get("load_best_model_at_end", True),
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        save_strategy="steps",
        dataloader_num_workers=cfg.get("dataloader_num_workers", 4),
        remove_unused_columns=False,
        run_name=cfg.get("run_name", "wechat-style-sft"),
        seed=cfg.get("seed", 42)
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
    print(f"预计训练步数: {cfg['max_steps']}")
    print(f"保存路径: {cfg['output_dir']}")
    
    try:
        trainer.train()
        
        # 保存模型
        trainer.save_model()
        print(f"模型已保存到: {cfg['output_dir']}")
        print("训练完成！")
        
        # 输出最终评估结果
        eval_results = trainer.evaluate()
        print(f"最终验证损失: {eval_results.get('eval_loss', 'N/A')}")
        
    except Exception as e:
        print(f"训练过程中出错: {e}")
        print("请检查显存是否足够，可尝试:")
        print("1. 使用更小的模型 (如 Qwen2.5-1.5B)")
        print("2. 减小 batch_size 或 max_seq_length")
        print("3. 检查CUDA环境是否正确")

if __name__ == "__main__":
    main()