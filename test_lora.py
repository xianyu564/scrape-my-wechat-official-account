#!/usr/bin/env python3
"""
LoRA 模型推理测试脚本
用于测试训练好的风格微调模型
"""

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel
import argparse

def load_model(base_model_name: str, lora_path: str):
    """加载基础模型和LoRA适配器"""
    print(f"正在加载基础模型: {base_model_name}")
    
    # 量化配置
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True, 
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True
    )
    
    # 加载tokenizer
    tokenizer = AutoTokenizer.from_pretrained(base_model_name, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # 加载基础模型
    model = AutoModelForCausalLM.from_pretrained(
        base_model_name, 
        quantization_config=bnb_config, 
        device_map="auto",
        torch_dtype=torch.float16
    )
    
    # 加载LoRA适配器
    print(f"正在加载LoRA适配器: {lora_path}")
    model = PeftModel.from_pretrained(model, lora_path)
    
    print("模型加载完成!")
    return model, tokenizer

def generate_text(model, tokenizer, prompt: str, max_new_tokens: int = 200, temperature: float = 0.7):
    """生成文本"""
    # 构建完整的输入
    system_prompt = "你是该公众号作者，保持其叙述节奏与转折。"
    full_prompt = f"<|system|>\n{system_prompt}\n<|user|>\n{prompt}\n<|assistant|>\n"
    
    print(f"输入提示: {prompt}")
    print("正在生成...")
    
    # 编码输入
    inputs = tokenizer(full_prompt, return_tensors="pt").to(model.device)
    
    # 生成文本
    with torch.no_grad():
        outputs = model.generate(
            **inputs, 
            max_new_tokens=max_new_tokens, 
            temperature=temperature,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id
        )
    
    # 解码输出
    generated = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    # 提取生成的部分
    if "<|assistant|>" in generated:
        result = generated.split("<|assistant|>")[-1].strip()
        if result.endswith("<|end|>"):
            result = result[:-7].strip()
    else:
        result = generated[len(full_prompt):].strip()
    
    return result

def compare_with_base(base_model_name: str, lora_path: str, prompt: str):
    """对比基础模型和LoRA模型的输出"""
    print("=== 对比基础模型和LoRA模型 ===\n")
    
    # 加载基础模型
    print("加载基础模型...")
    bnb_config = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4")
    tokenizer = AutoTokenizer.from_pretrained(base_model_name, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_name, 
        quantization_config=bnb_config, 
        device_map="auto"
    )
    
    # 生成基础模型输出
    print("\n--- 基础模型输出 ---")
    base_output = generate_text(base_model, tokenizer, prompt, max_new_tokens=150)
    print(base_output)
    
    # 加载LoRA模型
    print("\n加载LoRA模型...")
    lora_model = PeftModel.from_pretrained(base_model, lora_path)
    
    # 生成LoRA模型输出
    print("\n--- LoRA模型输出 ---")
    lora_output = generate_text(lora_model, tokenizer, prompt, max_new_tokens=150)
    print(lora_output)
    
    print("\n=== 对比完成 ===")

def interactive_mode(model, tokenizer):
    """交互模式"""
    print("\n进入交互模式，输入 'quit' 退出")
    print("=" * 50)
    
    while True:
        prompt = input("\n请输入提示词: ").strip()
        
        if prompt.lower() in ['quit', 'exit', '退出']:
            print("退出交互模式")
            break
        
        if not prompt:
            continue
        
        try:
            result = generate_text(model, tokenizer, prompt)
            print(f"\n生成结果:\n{result}")
            print("-" * 50)
        except Exception as e:
            print(f"生成时出错: {e}")

def main():
    parser = argparse.ArgumentParser(description="LoRA模型推理测试")
    parser.add_argument("--base_model", type=str, default="Qwen/Qwen2.5-7B-Instruct",
                       help="基础模型名称")
    parser.add_argument("--lora_path", type=str, default="outputs/qwen25-7b-sft-lora",
                       help="LoRA适配器路径")
    parser.add_argument("--prompt", type=str, 
                       help="测试提示词")
    parser.add_argument("--compare", action="store_true",
                       help="对比基础模型和LoRA模型")
    parser.add_argument("--interactive", action="store_true",
                       help="进入交互模式")
    
    args = parser.parse_args()
    
    # 默认测试提示词
    default_prompts = [
        "用我的口吻写一段关于早晨的感受",
        "以我的风格描述一次普通的购物经历",
        "用我的语气谈谈对当下生活节奏的看法",
        "以我的口吻写一段关于读书的心得",
        "用我的风格描述一个安静的傍晚"
    ]
    
    if args.compare:
        # 对比模式
        test_prompt = args.prompt or default_prompts[0]
        compare_with_base(args.base_model, args.lora_path, test_prompt)
    else:
        # 加载LoRA模型
        model, tokenizer = load_model(args.base_model, args.lora_path)
        
        if args.interactive:
            # 交互模式
            interactive_mode(model, tokenizer)
        else:
            # 单次测试
            if args.prompt:
                result = generate_text(model, tokenizer, args.prompt)
                print(f"\n生成结果:\n{result}")
            else:
                # 运行默认测试
                print("运行默认测试用例...")
                for i, prompt in enumerate(default_prompts, 1):
                    print(f"\n=== 测试 {i}/5 ===")
                    result = generate_text(model, tokenizer, prompt, max_new_tokens=150)
                    print(f"提示: {prompt}")
                    print(f"生成: {result}")
                    print("-" * 50)

if __name__ == "__main__":
    main()