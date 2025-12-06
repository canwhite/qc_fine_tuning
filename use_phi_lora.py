#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
使用微调后的 Phi-3.5-mini-instruct 模型进行推理
"""

import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import logging

# 设置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_finetuned_model(
    base_model_name="microsoft/Phi-3.5-mini-instruct",
    lora_adapter_path="./phi_checkpoint",
):
    """
    加载微调后的模型

    Args:
        base_model_name: 基础模型名称
        lora_adapter_path: LoRA适配器路径

    Returns:
        tuple: (model, tokenizer)
    """
    logger.info("正在加载tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(base_model_name, trust_remote_code=True)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.unk_token

    logger.info("正在加载基础模型...")
    # 加载基础模型
    model = AutoModelForCausalLM.from_pretrained(
        base_model_name,
        dtype=torch.float16,  # 使用新的 dtype 参数
        trust_remote_code=True,
        device_map=None,  # 不使用 device_map，手动处理设备
        attn_implementation="eager",  # 明确使用 eager attention
        low_cpu_mem_usage=True,
        # 添加以下参数以进一步减少警告
        use_cache=False,  # 避免缓存相关的警告
        torch_dtype=torch.float16,  # 确保数据类型一致
    )

    logger.info("正在加载LoRA适配器...")
    # 加载LoRA适配器
    try:
        # 加buff
        # 在上述model的基础上，再加上我们的训练数据，主义用的是PeftModel.from_pretrained
        model = PeftModel.from_pretrained(model, lora_adapter_path)
    except Exception as e:
        logger.error(f"加载LoRA适配器失败: {e}")
        logger.info("尝试使用不同方法加载...")
        # 加buff
        # 如果第一次失败，尝试不使用 strict 模式
        model = PeftModel.from_pretrained(model, lora_adapter_path, is_trainable=False)

    # 手动移动到合适的设备
    if torch.backends.mps.is_available():
        device = "mps"
    elif torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"

    model = model.to(device)
    logger.info(f"模型已移动到{device}设备")

    # 确保模型在评估模式
    model.eval()

    return model, tokenizer


def generate_response(model, tokenizer, prompt, max_new_tokens=512, temperature=0.7):
    """
    生成回复

    Args:
        model: 加载的模型
        tokenizer: 分词器
        prompt: 输入提示
        max_new_tokens: 最大生成token数
        temperature: 生成温度

    Returns:
        str: 生成的回复
    """
    try:
        # 构建对话格式
        messages = [{"role": "user", "content": prompt}]

        # 应用聊天模板
        text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

        logger.info(
            f"输入文本: {text[:100]}..." if len(text) > 100 else f"输入文本: {text}"
        )

        # 编码输入
        inputs = tokenizer(
            text, return_tensors="pt", padding=True, truncation=True, max_length=1024
        )

        # 将输入移动到模型所在设备
        device = next(model.parameters()).device
        inputs = {k: v.to(device) for k, v in inputs.items()}

        # 生成回复
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                do_sample=True,
                pad_token_id=(
                    tokenizer.pad_token_id
                    if tokenizer.pad_token is not None
                    else tokenizer.eos_token_id
                ),
                eos_token_id=tokenizer.eos_token_id,
                repetition_penalty=1.1,  # 减少重复
                use_cache=False,  # 关闭缓存以避免 DynamicCache 错误
            )

        # 解码输出
        full_response = tokenizer.decode(outputs[0], skip_special_tokens=True)

        logger.info(
            f"完整输出: {full_response[:200]}..."
            if len(full_response) > 200
            else f"完整输出: {full_response}"
        )

        # 提取生成的部分（去掉输入提示）
        if text in full_response:
            response = full_response.replace(text, "").strip()
        else:
            # 如果找不到完整的输入文本，尝试找到生成开始的位置
            response = full_response
            # 移除可能的重复输入部分
            if prompt in response:
                response = response.replace(prompt, "").strip()

        logger.info(
            f"生成回复: {response[:100]}..."
            if len(response) > 100
            else f"生成回复: {response}"
        )

        return response

    except Exception as e:
        logger.error(f"生成回复时出错: {e}")
        return f"生成失败: {str(e)}"


def interactive_chat(model, tokenizer):
    """
    交互式聊天
    """
    print("=== 与微调后的Phi模型聊天 ===")
    print("输入 'quit' 或 'exit' 退出")
    print("=" * 40)

    while True:
        user_input = input("\n你: ").strip()

        if user_input.lower() in ["quit", "exit", "退出"]:
            print("再见！")
            break

        if not user_input:
            continue

        print("AI: ", end="", flush=True)
        response = generate_response(model, tokenizer, user_input)
        print(response)


def test_examples(model, tokenizer):
    """
    测试一些示例问题
    """
    test_questions = [
        "写一个Python函数计算斐波那契数列",
        "将以下英文翻译成中文：The weather is really nice today.",
        "解释什么是机器学习",
        "写一封求职信，申请软件工程师职位",
    ]

    print("=== 测试示例 ===")
    for i, question in enumerate(test_questions, 1):
        print(f"\n问题 {i}: {question}")
        print("回答:", end=" ")

        response = generate_response(model, tokenizer, question, max_new_tokens=256)
        print(response)
        print("-" * 50)


def main():
    """
    主函数
    """
    try:
        # 检查必要的文件
        if not os.path.exists("./phi_checkpoint"):
            print("❌ 错误: 找不到 phi_checkpoint 目录")
            print("请确保在正确的项目目录中运行此脚本")
            return

        # 加载模型
        print("正在加载模型，这可能需要一些时间...")
        model, tokenizer = load_finetuned_model()

        print("\n🎉 模型加载成功！")
        print("模型信息:")
        print(f"  - 基础模型: microsoft/Phi-3.5-mini-instruct")
        print(f"  - LoRA适配器: ./phi_checkpoint")
        print(f"  - 设备: {next(model.parameters()).device}")

        print("\n选择使用方式:")
        print("1. 测试示例问题")
        print("2. 交互式聊天")

        choice = input("\n请选择 (1/2): ").strip()

        if choice == "1":
            test_examples(model, tokenizer)
        elif choice == "2":
            interactive_chat(model, tokenizer)
        else:
            print("无效选择，启动交互式聊天...")
            interactive_chat(model, tokenizer)

    except torch.cuda.OutOfMemoryError as e:
        print("❌ GPU内存不足！")
        print("建议:")
        print("1. 使用CPU模式运行")
        print("2. 减少其他占用GPU的程序")
        print("3. 使用更小的模型")
        logger.error(f"GPU内存不足: {e}")

    except Exception as e:
        logger.error(f"发生错误: {e}")
        print(f"\n❌ 错误: {e}")
        print("\n请检查:")
        print("1. 确保已安装所需依赖: uv add transformers torch peft datasets")
        print("2. 确保 phi_checkpoint 目录存在且包含必要的文件")
        print("3. 确保有足够的内存运行模型")
        print("4. 检查网络连接（首次运行需要下载模型）")

        # 提供更详细的错误信息
        if "cannot import" in str(e).lower():
            print("\n缺少依赖包，请运行:")
            print("uv add transformers torch peft datasets accelerate trl")
        elif "no such file" in str(e).lower():
            print("\n文件路径问题，请确保:")
            print("- 在正确的目录中运行脚本")
            print("- phi_checkpoint 目录存在")
        elif "out of memory" in str(e).lower():
            print("\n内存不足，请尝试:")
            print("- 关闭其他占用内存的程序")
            print("- 使用更小的批次大小")


if __name__ == "__main__":
    main()
