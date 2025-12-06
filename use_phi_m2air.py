#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
M2 Air专用版本 - 超轻量级Phi-3.5模型推理
专门针对8GB内存的M2 Air优化
"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import gc
import os

# 设置M2 Air优化环境变量
os.environ["PYTORCH_MPS_HIGH_WATERMARK_RATIO"] = "0.0"  # 禁用MPS内存高水位
os.environ["OMP_NUM_THREADS"] = "1"  # 限制CPU线程数

# 设置随机种子
torch.random.manual_seed(0)

# 清理内存
gc.collect()
if torch.backends.mps.is_available():
    torch.mps.empty_cache()


def load_model_for_m2air(model_name="microsoft/Phi-3.5-mini-instruct"):
    """
    为M2 Air加载超轻量级模型
    """
    print("🚀 正在为M2 Air加载超轻量级模型...")
    print("⚠️  这将占用约2-3GB内存，请确保关闭其他应用")

    # M2 Air极限优化配置
    model_kwargs = {
        "dtype": torch.float16,  # 使用半精度
        "trust_remote_code": True,
        "device_map": None,  # 手动控制设备
        "attn_implementation": "eager",  # 避免flash attention问题
        "low_cpu_mem_usage": True,
        "use_cache": False,  # 关闭缓存节省内存
        "max_memory": {"mps": "3GB"},  # M2 Air极限内存限制
        # 添加更多内存优化参数
        "torch_dtype": torch.float16,
        "offload_folder": "./offload_cache",  # 磁盘缓存
        "offload_state_dict": True,  # 卸载状态字典
    }

    try:
        # 第一步：只加载模型结构
        print("📦 步骤1: 加载模型结构...")
        model = AutoModelForCausalLM.from_pretrained(model_name, **model_kwargs)

        # 第二步：渐进式移动到设备
        print("📱 步骤2: 移动到MPS设备...")
        if torch.backends.mps.is_available():
            # 分块移动到MPS，避免内存峰值
            with torch.no_grad():
                model = model.to("mps")
            print("✅ 模型已移动到MPS")
        else:
            print("⚠️  MPS不可用，使用CPU（会更慢）")
            model = model.to("cpu")

        # 第三步：清理内存
        gc.collect()
        if torch.backends.mps.is_available():
            torch.mps.empty_cache()

        # 第四步：加载tokenizer
        print("📝 步骤3: 加载tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        # 第五步：不使用pipeline，直接返回模型和tokenizer
        print("🔧 步骤4: 准备直接生成模式...")
        print("✅ M2 Air专用模型加载完成！")
        return model, tokenizer

    except Exception as e:
        print(f"❌ M2 Air加载失败: {e}")
        print("🔄 尝试CPU模式...")

        # 降级到CPU模式
        model_kwargs_cpu = {
            "dtype": torch.float16,
            "trust_remote_code": True,
            "device_map": "cpu",
            "attn_implementation": "eager",
            "low_cpu_mem_usage": True,
            "use_cache": False,
        }

        model = AutoModelForCausalLM.from_pretrained(model_name, **model_kwargs_cpu)
        tokenizer = AutoTokenizer.from_pretrained(model_name)

        print("✅ CPU模式加载成功（会比较慢）")
        return model, tokenizer


def generate_response_m2air(
    model, tokenizer, messages, max_new_tokens=200, temperature=0.7
):
    """
    M2 Air优化的生成函数 - 直接使用model.generate()避免pipeline问题
    """
    try:
        # 获取用户消息
        if isinstance(messages, list) and len(messages) > 0:
            user_content = (
                messages[-1]["content"]
                if messages[-1]["role"] == "user"
                else str(messages)
            )
        else:
            user_content = str(messages)

        # 构建简单的输入格式
        if hasattr(tokenizer, "chat_template") and tokenizer.chat_template:
            text = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
        else:
            # 简单格式
            text = f"User: {user_content}\nAssistant: "

        print(f"🔤 输入: {text[:50]}...")

        # 编码输入
        inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=400)

        # 移动到模型设备
        device = next(model.parameters()).device
        inputs = {k: v.to(device) for k, v in inputs.items()}

        # 清理内存
        gc.collect()
        if torch.backends.mps.is_available():
            torch.mps.empty_cache()

        # 生成回复 - 最稳定配置
        with torch.no_grad():
            outputs = model.generate(
                input_ids=inputs["input_ids"],
                attention_mask=inputs.get("attention_mask", None),
                max_new_tokens=min(max_new_tokens, 150),  # 限制长度
                do_sample=temperature > 0.1,  # 根据温度决定是否采样
                temperature=temperature if temperature > 0.1 else 1.0,
                pad_token_id=(
                    tokenizer.pad_token_id
                    if tokenizer.pad_token_id
                    else tokenizer.eos_token_id
                ),
                eos_token_id=tokenizer.eos_token_id,
                use_cache=False,  # 关键：禁用缓存避免DynamicCache错误
            )

        # 解码输出
        full_text = tokenizer.decode(outputs[0], skip_special_tokens=True)

        # 提取生成的部分
        if text in full_text:
            response = full_text.replace(text, "").strip()
        else:
            # 备用提取方法
            response = (
                full_text.split("Assistant:")[-1].strip()
                if "Assistant:" in full_text
                else full_text.strip()
            )

        # 限制回复长度
        if len(response) > 300:
            response = response[:300] + "..."

        return response

    except Exception as e:
        print(f"⚠️  生成时出错: {e}")
        return f"生成失败: {str(e)}"


def simple_chat_m2air(model, tokenizer):
    """
    M2 Air专用简化聊天功能
    """
    print("\n=== M2 Air专用Phi聊天 ===")
    print("⚡ 超轻量级版本，专为8GB内存优化")
    print("💭 输入 'quit' 退出")
    print("=" * 40)

    # 初始化轻量对话历史
    messages = []
    max_history = 2  # 只保留最近2轮对话

    while True:
        try:
            user_input = input("\n你: ").strip()

            if user_input.lower() in ["quit", "exit", "退出", "q"]:
                print("👋 再见！")
                break

            if not user_input:
                continue

            # 限制输入长度
            if len(user_input) > 200:
                user_input = user_input[:200] + "..."
                print("⚠️  输入过长，已截断")

            # 构建消息格式
            messages = [{"role": "user", "content": user_input}]

            print("🤖 AI: ", end="", flush=True)
            response = generate_response_m2air(
                model, tokenizer, messages, max_new_tokens=150
            )
            print(response)

            # 添加助手回复并限制历史长度
            messages.append({"role": "assistant", "content": response})
            if len(messages) > max_history * 2:  # 每轮对话有用户和助手两条消息
                messages = messages[-max_history * 2 :]

            # 每轮对话后清理内存
            gc.collect()
            if torch.backends.mps.is_available():
                torch.mps.empty_cache()

        except KeyboardInterrupt:
            print("\n👋 用户中断，再见！")
            break
        except Exception as e:
            print(f"\n❌ 聊天错误: {e}")
            print("🔄 重置对话...")
            messages = []


def quick_test(model, tokenizer):
    """
    快速测试模型
    """
    print("\n=== 快速测试 ===")

    test_questions = [
        "Hello, how are you?",
        "用一句话介绍你自己",
        "2+2等于几？",
    ]

    for i, question in enumerate(test_questions, 1):
        print(f"\n问题{i}: {question}")
        print("回答: ", end="")

        messages = [{"role": "user", "content": question}]
        response = generate_response_m2air(
            model, tokenizer, messages, max_new_tokens=50, temperature=0.3
        )
        print(response)
        print("-" * 30)


def main():
    """
    M2 Air专用主函数
    """
    try:
        print("=" * 50)
        print("🍎 M2 Air专用Phi-3.5模型启动器")
        print("=" * 50)

        # 显示系统信息
        print(f"🔧 PyTorch版本: {torch.__version__}")
        print(f"📱 MPS可用: {torch.backends.mps.is_available()}")
        if torch.backends.mps.is_available():
            print(f"💾 MPS设备: {torch.backends.mps.is_built()}")

        # 加载模型
        model, tokenizer = load_model_for_m2air()

        # 显示模型信息
        device = next(model.parameters()).device
        print(f"\n📊 模型信息:")
        print(f"  - 设备: {device}")

        # 简化的参数计算
        total_params = sum(p.numel() for p in model.parameters())
        model_size_gb = total_params * 2 / 1024**3  # float16 = 2字节
        print(f"  - 参数量: {total_params//1000000}M")
        print(f"  - 理论大小: {model_size_gb:.1f}GB")

        print("\n🎮 选择模式:")
        print("1. 快速测试")
        print("2. 简单聊天")

        choice = input("请选择 (1/2): ").strip()

        if choice == "1":
            quick_test(model, tokenizer)
        else:
            simple_chat_m2air(model, tokenizer)

    except Exception as e:
        print(f"\n❌ 启动失败: {e}")
        print("\n💡 建议:")
        print("1. 关闭其他应用程序释放内存")
        print("2. 重启电脑清理内存")
        print("3. 确保有足够的磁盘空间用于缓存")

        # 提供降级建议
        if "out of memory" in str(e).lower():
            print("4. 考虑使用更小的模型")
            print("5. 重启进入安全模式释放内存")


if __name__ == "__main__":
    main()
