#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
测试修复后的微调模型
"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import warnings

warnings.filterwarnings("ignore")

# 设置环境变量
import os

os.environ["PYTORCH_MPS_HIGH_WATERMARK_RATIO"] = "0.0"
os.environ["OMP_NUM_THREADS"] = "1"


def load_fixed_model():
    """加载修复后的模型"""
    print("🔄 加载修复后的微调模型...")

    base_model_name = "Qwen/Qwen2.5-0.5B-Instruct"
    lora_path = "./qwen_fixed_finetuned"

    try:
        # 加载tokenizer
        tokenizer = AutoTokenizer.from_pretrained(lora_path)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        # 加载基础模型
        base_model = AutoModelForCausalLM.from_pretrained(
            base_model_name,
            torch_dtype=torch.float16,
            trust_remote_code=True,
        )

        # 加载LoRA适配器
        model = PeftModel.from_pretrained(base_model, lora_path)

        # 移动到MPS
        if torch.backends.mps.is_available():
            model = model.to("mps")
            print("✅ 模型已移动到MPS")
        else:
            print("💻 使用CPU")

        return model, tokenizer

    except Exception as e:
        print(f"❌ 模型加载失败: {e}")
        return None, None


def generate_response(model, tokenizer, prompt):
    """生成回复"""
    try:
        # 构建消息格式
        messages = [{"role": "user", "content": prompt}]
        text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

        # 编码输入
        inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=256)

        # 移动到设备
        device = next(model.parameters()).device
        inputs = {k: v.to(device) for k, v in inputs.items()}

        # 生成回复
        with torch.no_grad():
            outputs = model.generate(
                input_ids=inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
                max_new_tokens=100,
                do_sample=True,
                temperature=0.7,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )

        # 解码并提取回复
        full_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        response = full_text.replace(text, "").strip()

        return response

    except Exception as e:
        return f"生成失败: {str(e)}"


def main():
    """主函数"""
    print("=" * 50)
    print("🧪 测试修复后的Qwen2.5微调模型")
    print("=" * 50)

    # 加载模型
    model, tokenizer = load_fixed_model()

    if model is None:
        print("❌ 模型加载失败，程序退出")
        return

    # 测试问题
    test_prompts = [
        "用Python写一个计算圆面积的函数",
        "1+1等于几？",
        "解释什么是量子计算",
        "写一个简单的Flask应用",
    ]

    print(f"\n🚀 开始测试（共{len(test_prompts)}个问题）...")

    for i, prompt in enumerate(test_prompts, 1):
        print(f"\n{'='*40}")
        print(f"📝 测试 {i}: {prompt}")
        print("-" * 40)

        try:
            print("🤖 微调模型回答:")
            response = generate_response(model, tokenizer, prompt)
            print(f"   {response}")

        except Exception as e:
            print(f"❌ 生成失败: {e}")

    print(f"\n✅ 测试完成！")


if __name__ == "__main__":
    main()
