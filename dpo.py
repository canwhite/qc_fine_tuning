"""
DPO (Direct Preference Optimization) 训练示例
==========================================
这个脚本展示如何使用 DPO 算法进行偏好学习

核心概念：
- DPO vs PPO 的区别：
  * PPO: 在线训练，需要 Reward Model，需要 ValueHead
  * DPO: 离线训练，使用偏好数据对，不需要 Reward Model 和 ValueHead

- DPO 的核心：
  * 输入: (prompt, chosen, rejected) 三元组
  * 目标: 让模型更倾向于生成 chosen 而非 rejected
  * 不需要显式的奖励模型，直接优化偏好

运行前安装: pip install trl transformers torch
"""

import torch
from torch.utils.data import Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import DPOConfig, DPOTrainer

# ============================================================
# 第一部分：偏好数据集
# ============================================================


class PreferenceDataset(Dataset):
    """
    DPO 训练数据集

    DPO 需要的是 (prompt, chosen, rejected) 三元组
    - prompt: 问题或提示
    - chosen: 人类选择的更好回答
    - rejected: 人类拒绝的较差回答

    这与 PPO 不同：
    - PPO: 只有 prompt，回答由模型生成
    - DPO: 有 prompt + chosen + rejected，离线数据
    """

    def __init__(self, data):
        """
        Args:
            data: List of dict, 每个 dict 包含:
                  - prompt: str
                  - chosen: str
                  - rejected: str
        """
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


# 示例偏好数据
# 注意：这些是人工构造的示例，真实场景需要人类标注
PREFERENCE_DATA = [
    {
        "prompt": "什么是机器学习？",
        "chosen": "机器学习是人工智能的一个分支，它使计算机能够从数据中学习和改进，而无需显式编程。主要包括监督学习、无监督学习和强化学习等方法。",
        "rejected": "机器学习就是让机器学习。",
    },
    {
        "prompt": "如何学习编程？",
        "chosen": "建议从以下步骤开始：1. 选择一门适合初学者的语言如 Python；2. 学习基础语法和数据结构；3. 多做练习项目；4. 阅读优秀代码；5. 参与开源项目实践。",
        "rejected": "随便学学就行了。",
    },
    {
        "prompt": "什么是深度学习？",
        "chosen": "深度学习是机器学习的子领域，使用多层神经网络来学习数据的层次化表示。它在图像识别、自然语言处理和语音识别等领域取得了显著成果。",
        "rejected": "深度学习就是很深的学习。",
    },
    {
        "prompt": "推荐一些编程语言。",
        "chosen": "根据不同用途推荐：\n1. Python - 适合初学者和数据科学\n2. JavaScript - 前端开发首选\n3. Java - 企业级应用\n4. Go - 云服务和微服务\n5. Rust - 系统编程",
        "rejected": "推荐 C++。",
    },
    {
        "prompt": "什么是神经网络？",
        "chosen": "神经网络是一种模拟人脑神经元连接的计算模型，由输入层、隐藏层和输出层组成。每个神经元接收输入，通过激活函数处理后输出，通过训练调整连接权重来学习模式。",
        "rejected": "神经网络就是网络。",
    },
    {
        "prompt": "如何提高编程能力？",
        "chosen": "提高编程能力的有效方法：\n1. 坚持每天编码练习\n2. 阅读和理解优秀项目源码\n3. 参与开源项目贡献\n4. 学习设计模式和最佳实践\n5. 解决实际项目问题\n6. 接受代码审查反馈",
        "rejected": "多写代码就行。",
    },
    {
        "prompt": "什么是自然语言处理？",
        "chosen": "自然语言处理（NLP）是人工智能的重要分支，研究计算机如何理解、解释和生成人类语言。应用包括机器翻译、情感分析、问答系统、文本生成等。",
        "rejected": "NLP 就是处理语言。",
    },
    {
        "prompt": "解释一下算法。",
        "chosen": "算法是解决特定问题的有限步骤序列。好的算法应该具有：\n1. 正确性 - 得到正确结果\n2. 可读性 - 易于理解\n3. 健壮性 - 处理异常输入\n4. 高效性 - 时间和空间复杂度合理",
        "rejected": "算法就是步骤。",
    },
]


# ============================================================
# 第二部分：DPO 训练主流程
# ============================================================


def main():
    print("=" * 60)
    print("DPO (Direct Preference Optimization) 训练演示")
    print("=" * 60)

    # ---------- 2.1 配置 ----------
    print("\n[1/6] 加载配置...")

    config = DPOConfig(
        # 模型相关
        model_name_or_path="Qwen/Qwen2.5-0.5B-Instruct",  # 小模型，适合演示
        # 训练超参数
        learning_rate=5e-7,  # DPO 通常用较小的学习率
        per_device_train_batch_size=2,  # 批大小
        gradient_accumulation_steps=1,
        # DPO 特有参数
        beta=0.1,  # KL 散度惩罚系数（控制模型偏离参考模型的程度）
        # beta 越大，模型越保守（更接近参考模型）
        # beta 越小，模型越激进（更倾向于偏好数据）
        # 其他
        max_length=512,  # 最大序列长度
        max_prompt_length=128,  # 最大 prompt 长度
        seed=42,
        output_dir="./dpo_output",
        logging_steps=1,
        save_strategy="no",  # 演示用，不保存 checkpoint
        report_to="none",  # 可用 "wandb" 或 "tensorboard"
    )

    print(f"   模型: {config.model_name_or_path}")
    print(f"   学习率: {config.learning_rate}")
    print(f"   Beta (KL 系数): {config.beta}")

    # ---------- 2.2 加载 Tokenizer ----------
    print("\n[2/6] 加载 Tokenizer...")

    tokenizer = AutoTokenizer.from_pretrained(
        config.model_name_or_path,
        trust_remote_code=True,
    )

    # 设置 padding token
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print(f"   词表大小: {len(tokenizer)}")

    # ---------- 2.3 加载模型 ----------
    print("\n[3/6] 加载模型...")

    # Policy 模型（会更新）
    model = AutoModelForCausalLM.from_pretrained(
        config.model_name_or_path,
        trust_remote_code=True,
        torch_dtype=torch.float16,
    )

    # 移动到设备
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    model = model.to(device)
    print(f"   设备: {device}")

    # Reference 模型（冻结）
    # DPO 需要参考模型来计算 KL 散度
    # 注意：DPO 不需要 ValueHead！
    ref_model = AutoModelForCausalLM.from_pretrained(
        config.model_name_or_path,
        trust_remote_code=True,
        torch_dtype=torch.float16,
    )
    ref_model = ref_model.to(device)
    ref_model.eval()  # 设为评估模式，冻结参数

    print("   已创建参考模型（冻结）")
    print("\n   DPO vs PPO 模型区别:")
    print("   - DPO: 不需要 ValueHead，只用 LM Head")
    print("   - PPO: 需要 ValueHead 预测 value")

    # ---------- 2.4 初始化组件 ----------
    print("\n[4/6] 初始化训练组件...")

    # 偏好数据集
    dataset = PreferenceDataset(PREFERENCE_DATA)
    print(f"   数据集大小: {len(dataset)}")
    print("   数据格式: (prompt, chosen, rejected)")

    # DPO Trainer
    dpo_trainer = DPOTrainer(
        model=model,
        ref_model=ref_model,
        args=config,
        train_dataset=dataset,
        tokenizer=tokenizer,
    )
    print("   DPO Trainer 已初始化")

    # ---------- 2.5 训练 ----------
    print("\n[5/6] 开始 DPO 训练...")
    print("-" * 60)

    # DPO 训练过程
    # 注意：DPO 不需要在线生成！直接使用离线数据
    print("\n   DPO 训练流程:")
    print("   1. 对每个样本：(prompt, chosen, rejected)")
    print("   2. 计算 chosen 和 rejected 的对数概率")
    print("   3. 计算 DPO Loss:")
    print("      Loss = -log(sigmoid(beta * (log_prob_chosen - log_prob_rejected)))")
    print("   4. 反向传播更新 Policy 模型")
    print("   5. Reference 模型保持不变")
    print()

    # 开始训练
    train_result = dpo_trainer.train()

    # 打印训练统计
    print("\n" + "=" * 60)
    print("训练完成！")
    print("=" * 60)
    print(f"\n   训练步数: {train_result.global_step}")
    print(f"   训练损失: {train_result.training_loss:.4f}")

    # ---------- 2.6 测试训练后的模型 ----------
    print("\n[6/6] 测试训练后的模型...")
    print("-" * 60)

    test_prompts = [
        "什么是深度学习？",
        "如何提高编程能力？",
    ]

    model.eval()
    with torch.no_grad():
        for prompt in test_prompts:
            # 编码
            inputs = tokenizer(prompt, return_tensors="pt").to(device)

            # 生成
            outputs = model.generate(
                **inputs,
                max_new_tokens=100,
                do_sample=True,
                temperature=0.7,
                top_p=0.9,
                pad_token_id=tokenizer.pad_token_id,
            )

            # 解码
            response = tokenizer.decode(outputs[0], skip_special_tokens=True)
            response = response[len(prompt) :]  # 去掉 prompt 部分

            print(f"\n  Q: {prompt}")
            print(f"  A: {response}")

    # ---------- 保存模型 ----------
    print("\n" + "=" * 60)

    # 保存模型
    save_path = "./dpo_finetuned_model"
    model.save_pretrained(save_path)
    tokenizer.save_pretrained(save_path)
    print(f"\n模型已保存到: {save_path}")


# ============================================================
# 附录：DPO vs PPO 详细对比
# ============================================================


def explain_dpo_vs_ppo():
    """
    DPO 与 PPO 的详细对比
    """
    print("""
    ╔══════════════════════════════════════════════════════════╗
    ║              DPO vs PPO 详细对比                         ║
    ╠══════════════════════════════════════════════════════════╣
    ║                                                          ║
    ║  1. 训练数据                                             ║
    ║     PPO: 只有 prompt，回答由模型实时生成                  ║
    ║     DPO: (prompt, chosen, rejected) 偏好对数据           ║
    ║                                                          ║
    ║  2. 训练方式                                             ║
    ║     PPO: 在线训练 (on-policy)，需要采样                  ║
    ║     DPO: 离线训练 (off-policy)，不需要采样               ║
    ║                                                          ║
    ║  3. 模型架构                                             ║
    ║     PPO: Policy + ValueHead + Reference + Reward Model   ║
    ║     DPO: Policy + Reference (不需要 ValueHead)           ║
    ║                                                          ║
    ║  4. 奖励信号                                             ║
    ║     PPO: 需要训练好的 Reward Model                       ║
    ║     DPO: 不需要显式奖励模型，直接从偏好学习              ║
    ║                                                          ║
    ║  5. 训练稳定性                                           ║
    ║     PPO: 较复杂，需要调节多个超参数                      ║
    ║     DPO: 更稳定，超参数较少                              ║
    ║                                                          ║
    ║  6. 计算成本                                             ║
    ║     PPO: 较高（需要在线生成 + 训练多个模型）             ║
    ║     DPO: 较低（离线训练，只需要一个模型）                ║
    ║                                                          ║
    ║  7. 适用场景                                             ║
    ║     PPO: 在线学习，需要实时反馈的场景                    ║
    ║     DPO: 有大量偏好数据的场景，离线对齐                  ║
    ║                                                          ║
    ║  8. DPO 损失函数                                         ║
    ║                                                          ║
    ║     L_DPO = -E[log σ(β (log π(y_chosen|x)                ║
    ║                         - log π(y_rejected|x)))]         ║
    ║                                                          ║
    ║     其中：                                               ║
    ║     - π: Policy 模型概率                                 ║
    ║     - β: KL 惩罚系数                                     ║
    ║     - σ: Sigmoid 函数                                    ║
    ║                                                          ║
    ║  9. 为什么 DPO 有效？                                    ║
    ║                                                          ║
    ║     DPO 证明了：在 Bradley-Terry 偏好模型下，            ║
    ║     最优策略可以用显式公式表示，不需要训练奖励模型       ║
    ║                                                          ║
    ╚══════════════════════════════════════════════════════════╝
    """)


if __name__ == "__main__":
    # 打印对比解释
    explain_dpo_vs_ppo()

    # 运行训练
    main()
