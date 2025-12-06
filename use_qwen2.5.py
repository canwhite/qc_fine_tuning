#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Qwen2.5-0.5B-Instruct 模型调用案例
专为M2 Air优化的超轻量级版本（仅0.5B参数）
"""

import sys
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import gc
import os
import warnings

# 忽略警告
warnings.filterwarnings("ignore")

# 设置M2 Air优化环境变量
os.environ["PYTORCH_MPS_HIGH_WATERMARK_RATIO"] = "0.0"
os.environ["OMP_NUM_THREADS"] = "1"

# 设置随机种子
torch.random.manual_seed(0)


def load_qwen3_model(model_name="Qwen/Qwen2.5-0.5B-Instruct"):
    """
    加载Qwen2.5-0.5B-Instruct模型
    """
    print("🚀 加载Qwen2.5-0.5B-Instruct模型...")
    print("💡 这是Qwen2.5系列的0.5B超轻量级模型，内存占用极小")

    # Qwen3专用配置 - 极度优化内存
    model_kwargs = {
        "torch_dtype": torch.float16,
        "device_map": None,  # 手动控制
        "trust_remote_code": True,
        "use_cache": False,  # 避免缓存问题
        "low_cpu_mem_usage": True,
        "max_memory": {"mps": "2GB"},  # 严格限制内存
    }

    try:
        # 加载模型
        print("📦 加载超轻量级模型...")
        model = AutoModelForCausalLM.from_pretrained(model_name, **model_kwargs)

        # 移动到MPS
        if torch.backends.mps.is_available():
            model = model.to("mps")
            print("✅ 模型已移动到MPS")
        else:
            model = model.to("cpu")
            print("💻 使用CPU")

        # 加载tokenizer
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        print(f"🎉 Qwen2.5-0.5B加载完成！")
        return model, tokenizer

    except Exception as e:
        print(f"❌ 加载失败: {e}")
        raise


def generate_qwen3_response(
    model, tokenizer, prompt, max_new_tokens=200, temperature=0.7
):
    """
    使用Qwen3生成回复
    """
    try:
        # Qwen3使用标准的chat template
        messages = [{"role": "user", "content": prompt}]

        # 应用聊天模板
        text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

        # 编码输入 - 限制输入长度
        inputs = tokenizer(
            text, return_tensors="pt", truncation=True, max_length=256  # 限制输入长度
        )

        # 移动到设备
        device = next(model.parameters()).device
        inputs = {k: v.to(device) for k, v in inputs.items()}

        # 清理内存
        gc.collect()
        if torch.backends.mps.is_available():
            torch.mps.empty_cache()

        # 生成回复 - 极度保守的配置
        with torch.no_grad():
            outputs = model.generate(
                input_ids=inputs["input_ids"],
                attention_mask=inputs.get("attention_mask", None),
                max_new_tokens=min(max_new_tokens, 150),  # 更短的回复
                do_sample=temperature > 0.2,  # 减少采样
                temperature=temperature if temperature > 0.2 else 1.0,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
                use_cache=False,
                repetition_penalty=1.05,  # 避免重复
            )

        # 解码
        full_text = tokenizer.decode(outputs[0], skip_special_tokens=True)

        # 提取回复
        response = full_text.replace(text, "").strip()

        # 限制回复长度
        if len(response) > 300:
            response = response[:300] + "..."

        return response

    except Exception as e:
        return f"生成失败: {str(e)}"


def qwen3_chat(model, tokenizer):
    """
    Qwen3聊天功能
    """
    print("\n=== Qwen2.5-0.5B 聊天 ===")
    print("🤖 阿里巴巴Qwen2.5系列，0.5B超轻量级")
    print("💭 输入 'quit' 退出 | 'clear' 清除历史")
    print("=" * 45)

    while True:
        try:
            user_input = input("\n你: ").strip()

            if user_input.lower() in ["quit", "exit", "退出", "q"]:
                print("👋 再见!")
                break

            if user_input.lower() in ["clear", "清除", "清空"]:
                print("🧹 对话历史已清除")
                continue

            if not user_input:
                continue

            print("🤖 Qwen3: ", end="", flush=True)
            response = generate_qwen3_response(model, tokenizer, user_input)
            print(response)

        except KeyboardInterrupt:
            print("\n👋 用户中断")
            break
        except Exception as e:
            print(f"❌ 错误: {e}")


def qwen3_test(model, tokenizer):
    """
    测试Qwen3功能
    """
    print("\n=== Qwen2.5-0.5B 功能测试 ===")

    test_cases = [
        ("编程", "用Python写一个hello world"),
        ("翻译", "将'你好世界'翻译成英文"),
        ("数学", "7 + 8 等于多少？"),
        ("常识", "1+1等于几？"),
        ("创意", "用三个词描述春天"),
    ]

    for category, question in test_cases:
        print(f"\n📝 [{category}] {question}")
        print("回答: ", end="")

        response = generate_qwen3_response(
            model, tokenizer, question, max_new_tokens=80, temperature=0.3
        )
        print(response)
        print("-" * 40)


def model_comparison():
    """
    模型对比
    """
    print("\n=== 模型参数对比 ===")
    print("📊 内存使用估算:")
    print("-" * 30)
    print("Qwen2.5-0.5B-Instruct:")
    print("  🔹 参数: 0.5B")
    print("  🔹 内存: ~1GB (float16)")
    print("  🔹 特点: 最新架构，极轻量")
    print()
    print("Phi-3.5-mini:")
    print("  🔹 参数: 3.8B")
    print("  🔹 内存: ~7.6GB (float16)")
    print("  🔹 特点: 微软开发，能力较强")
    print()
    print("Qwen2.5-1.5B-Instruct:")
    print("  🔹 参数: 1.5B")
    print("  🔹 内存: ~3GB (float16)")
    print("  🔹 特点: 平衡性能和大小")


def memory_info():
    """
    内存信息
    """
    print("\n=== M2 Air 内存建议 ===")
    print("💡 8GB M2 Air 使用建议:")
    print("1. 关闭浏览器标签页")
    print("2. 关闭其他大型应用")
    print("3. 使用Activity Monitor监控内存")
    print("4. 如果内存不够，重启电脑")


def main():
    """
    主函数
    """
    try:
        print("=" * 50)
        print("🚀 Qwen2.5-0.5B-Instruct 启动器")
        print("=" * 50)

        # 系统信息
        print(f"Python: {sys.version}")
        print(f"PyTorch: {torch.__version__}")
        print(f"MPS: {torch.backends.mps.is_available()}")

        # 模型信息
        model_comparison()
        memory_info()

        # 加载模型
        model, tokenizer = load_qwen3_model()

        # 显示实际模型信息
        device = next(model.parameters()).device
        total_params = sum(p.numel() for p in model.parameters())
        model_size_gb = total_params * 2 / 1024**3

        print(f"\n📊 实际模型信息:")
        print(f"  📍 设备: {device}")
        print(f"  🔢 参数: {total_params//1000000}M")
        print(f"  💾 理论大小: {model_size_gb:.1f}GB")
        print(f"  ⚡ 适合M2 Air: {'✅' if model_size_gb < 2 else '❌'}")

        # 选择功能
        print("\n🎮 选择功能:")
        print("1. 基础功能测试")
        print("2. 自由聊天")
        print("3. 查看模型信息")

        choice = input("请选择 (1/2/3): ").strip()

        if choice == "1":
            qwen3_test(model, tokenizer)
        elif choice == "2":
            qwen3_chat(model, tokenizer)
        elif choice == "3":
            model_comparison()
            print(f"\n当前模型:")
            print(f"  - Qwen2.5-0.5B")
            print(f"  - 参数: {total_params//1000000}M")
            print(f"  - 内存: {model_size_gb:.1f}GB")
            print(f"  - 设备: {device}")
        else:
            print("启动聊天模式...")
            qwen3_chat(model, tokenizer)

    except Exception as e:
        print(f"❌ 启动失败: {e}")
        print("\n💡 建议:")
        print("1. 检查网络连接")
        print("2. 安装transformers: pip install transformers")
        print("3. 关闭其他应用释放内存")
        print("4. 重启M2 Air")


if __name__ == "__main__":
    main()
