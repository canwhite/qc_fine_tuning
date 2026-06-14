"""
PPO + RLHF 完整训练示例（使用真实 Reward Model）
================================================
这个脚本展示如何使用 PPO 算法进行 RLHF 训练
使用 HuggingFace 上的真实 Reward Model 替代模拟

核心概念：
- Policy: 生成文本的语言模型（带 ValueHead）
- ValueHead: 预测未来累积奖励
- Reward Model: 真实的奖励模型，从 HuggingFace 加载
- PPO: 强化学习算法，用 advantage 更新策略

运行前安装: pip install trl transformers torch
"""

from typing import List

import torch
from torch.utils.data import DataLoader, Dataset
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
)
from trl import (
    AutoModelForCausalLMWithValueHead,
    PPOConfig,
    PPOTrainer,
    create_reference_model,
)

# ============================================================
# 第一部分：真实 Reward Model（从 HuggingFace 加载）
# ============================================================


class RealRewardModel:
    """
    真实的 Reward Model

    从 HuggingFace 加载预训练的奖励模型
    输入: 问题 + 回答
    输出: 标量奖励分数（越高越好）

    可用的开源 Reward Model:
    - OpenAssistant/reward-model-deberta-v3-large-v2 (英文)
    - OpenAssistant/reward-model-deberta-v3-base (英文，较小)
    - Ray2333/reward-model-reward-model-english-chinese (中英文)
    """

    def __init__(
        self,
        model_name: str = "OpenAssistant/reward-model-deberta-v3-large-v2",
        device: str = "cpu",
    ):
        print(f"   加载 Reward Model: {model_name}")

        self.device = device
        self.model = AutoModelForSequenceClassification.from_pretrained(
            model_name,
            torch_dtype=torch.float16 if device != "cpu" else torch.float32,
        )
        self.model = self.model.to(device)
        self.model.eval()

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

        print(f"   Reward Model 设备: {device}")

    def __call__(self, query: str, response: str) -> float:
        """
        计算奖励分数

        Args:
            query: 问题
            response: 模型生成的回答

        Returns:
            奖励分数（标量）
        """
        # 构造输入格式
        # 不同 Reward Model 的输入格式可能不同
        # OpenAssistant 格式: "Human: query\n\nAssistant: response"
        input_text = f"Human: {query}\n\nAssistant: {response}"

        # Tokenize
        inputs = self.tokenizer(
            input_text,
            return_tensors="pt",
            truncation=True,
            max_length=512,
            padding=True,
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        # 前向传播
        with torch.no_grad():
            outputs = self.model(**inputs)
            # 获取奖励分数（logits 的第一个元素）
            reward = outputs.logits[0, 0].item()

        return reward

    def batch_call(self, queries: List[str], responses: List[str]) -> List[float]:
        """
        批量计算奖励分数（更高效）

        Args:
            queries: 问题列表
            responses: 回答列表

        Returns:
            奖励分数列表
        """
        # 构造输入
        input_texts = [
            f"Human: {q}\n\nAssistant: {r}" for q, r in zip(queries, responses)
        ]

        # Tokenize
        inputs = self.tokenizer(
            input_texts,
            return_tensors="pt",
            truncation=True,
            max_length=512,
            padding=True,
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        # 前向传播
        with torch.no_grad():
            outputs = self.model(**inputs)
            rewards = outputs.logits.squeeze(-1).tolist()

        return rewards


# ============================================================
# 第二部分：训练数据集
# ============================================================


class PPODataset(Dataset):
    """
    PPO 训练数据集

    PPO 需要的是"问题"，回答由模型实时生成
    """

    def __init__(self, queries: List[str]):
        self.queries = queries

    def __len__(self):
        return len(self.queries)

    def __getitem__(self, idx):
        return {"query": self.queries[idx]}


TRAINING_QUERIES = [
    "What is artificial intelligence?",
    "Please introduce machine learning.",
    "What are the applications of deep learning?",
    "How to learn programming?",
    "What is natural language processing?",
    "Please recommend some programming languages.",
    "What is a neural network?",
    "How to improve programming skills?",
    "What is data science?",
    "Please explain algorithms.",
]


# ============================================================
# 第三部分：PPO 训练主流程
# ============================================================


def main():
    print("=" * 60)
    print("PPO + RLHF 训练演示（使用真实 Reward Model）")
    print("=" * 60)

    device = "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"\n设备: {device}")

    # ---------- 3.1 配置 ----------
    print("\n[1/7] 加载配置...")

    config = PPOConfig(
        model_name="Qwen/Qwen2.5-0.5B-Instruct",
        learning_rate=1e-5,
        batch_size=2,
        mini_batch_size=2,
        gradient_accumulation_steps=1,
        ppo_epochs=4,
        gamma=1.0,
        lam=0.95,
        adapt_kl_ctrl=True,
        init_kl_coef=0.2,
        target_kl=0.1,
        seed=42,
        log_with=None,
    )

    print(f"   Policy 模型: {config.model_name}")
    print(f"   学习率: {config.learning_rate}")
    print(f"   批大小: {config.batch_size}")

    # ---------- 3.2 加载 Tokenizer ----------
    print("\n[2/7] 加载 Tokenizer...")

    tokenizer = AutoTokenizer.from_pretrained(
        config.model_name,
        trust_remote_code=True,
    )

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    print(f"   词表大小: {len(tokenizer)}")

    # ---------- 3.3 加载 Policy 模型 ----------
    print("\n[3/7] 加载 Policy 模型（带 ValueHead）...")

    model = AutoModelForCausalLMWithValueHead.from_pretrained(
        config.model_name,
        trust_remote_code=True,
        torch_dtype=torch.float16,
    )
    model = model.to(device)

    print(f"   设备: {device}")

    ref_model = create_reference_model(model)
    print("   已创建参考模型（冻结）")

    print("\n   模型结构:")
    print(f"   - 主干: {model.pretrained_model.__class__.__name__}")
    print(f"   - LM Head: 输出 logits (词表大小: {len(tokenizer)})")
    print(f"   - Value Head: 输出标量 value")

    # ---------- 3.4 加载 Reward Model ----------
    print("\n[4/7] 加载 Reward Model...")

    reward_model = RealRewardModel(
        model_name="OpenAssistant/reward-model-deberta-v3-large-v2",
        device=device,
    )
    print("   奖励模型: OpenAssistant/reward-model-deberta-v3-large-v2")

    # ---------- 3.5 初始化组件 ----------
    print("\n[5/7] 初始化训练组件...")

    dataset = PPODataset(TRAINING_QUERIES)
    dataloader = DataLoader(dataset, batch_size=config.batch_size, shuffle=True)
    print(f"   数据集大小: {len(dataset)}")

    ppo_trainer = PPOTrainer(
        config=config,
        model=model,
        ref_model=ref_model,
        tokenizer=tokenizer,
    )
    print("   PPO Trainer 已初始化")

    # ---------- 3.6 训练循环 ----------
    print("\n[6/7] 开始 PPO 训练...")
    print("-" * 60)

    generation_kwargs = {
        "max_new_tokens": 50,
        "min_new_tokens": 10,
        "do_sample": True,
        "temperature": 0.7,
        "top_k": 50,
        "top_p": 0.9,
        "pad_token_id": tokenizer.pad_token_id,
    }

    num_epochs = 3

    for epoch in range(num_epochs):
        epoch_stats = {"loss": [], "reward": [], "kl": [], "entropy": []}

        for batch_idx, batch in enumerate(dataloader):
            queries = batch["query"]

            # Step 1: 编码查询
            query_tensors = []
            for q in queries:
                tokens = tokenizer(q, return_tensors="pt", padding=False)["input_ids"]
                query_tensors.append(tokens.squeeze(0).to(device))

            # Step 2: 生成回答
            response_tensors = ppo_trainer.generate(
                query_tensors,
                return_prompt=False,
                **generation_kwargs,
            )

            # Step 3: 解码回答
            responses = []
            for r in response_tensors:
                text = tokenizer.decode(r, skip_special_tokens=True)
                responses.append(text)

            # Step 4: 计算奖励（使用真实 Reward Model）
            rewards = []
            for query, response in zip(queries, responses):
                score = reward_model(query, response)
                reward_tensor = torch.tensor(score, dtype=torch.float32).to(device)
                rewards.append(reward_tensor)

            # Step 5: PPO 训练步骤
            stats = ppo_trainer.step(
                queries=query_tensors,
                responses=response_tensors,
                scores=rewards,
            )

            epoch_stats["loss"].append(stats["ppo/loss/total"])
            epoch_stats["reward"].append(stats["ppo/mean_scores"])
            epoch_stats["kl"].append(stats["ppo/mean_non_score_reward"])
            epoch_stats["entropy"].append(stats["ppo/mean_entropy"])

            if (batch_idx + 1) % 1 == 0:
                print(
                    f"\n  Epoch {epoch + 1}/{num_epochs} | Batch {batch_idx + 1}/{len(dataloader)}"
                )
                print(f"  ├─ Query: {queries[0][:30]}...")
                print(f"  ├─ Response: {responses[0][:50]}...")
                print(f"  ├─ Reward: {rewards[0].item():.3f}")
                print(f"  ├─ Loss: {stats['ppo/loss/total']:.4f}")
                print(f"  ├─ KL Div: {stats['ppo/mean_non_score_reward']:.4f}")
                print(f"  └─ Entropy: {stats['ppo/mean_entropy']:.4f}")

        print(f"\n  {'=' * 50}")
        print(f"  Epoch {epoch + 1} 总结:")
        print(
            f"  ├─ 平均 Loss: {sum(epoch_stats['loss']) / len(epoch_stats['loss']):.4f}"
        )
        print(
            f"  ├─ 平均 Reward: {sum(epoch_stats['reward']) / len(epoch_stats['reward']):.4f}"
        )
        print(f"  ├─ 平均 KL: {sum(epoch_stats['kl']) / len(epoch_stats['kl']):.4f}")
        print(
            f"  └─ 平均 Entropy: {sum(epoch_stats['entropy']) / len(epoch_stats['entropy']):.4f}"
        )
        print(f"  {'=' * 50}")

    # ---------- 3.7 测试训练后的模型 ----------
    print("\n[7/7] 测试训练后的模型...")
    print("-" * 60)

    test_queries = [
        "What is deep learning?",
        "How to start learning programming?",
    ]

    model.eval()
    with torch.no_grad():
        for query in test_queries:
            inputs = tokenizer(query, return_tensors="pt").to(device)

            outputs = model.generate(
                **inputs,
                max_new_tokens=80,
                do_sample=True,
                temperature=0.8,
                top_p=0.9,
                pad_token_id=tokenizer.pad_token_id,
            )

            response = tokenizer.decode(outputs[0], skip_special_tokens=True)
            response = response[len(query) :]

            print(f"\n  Q: {query}")
            print(f"  A: {response}")

            reward = reward_model(query, response)
            print(f"  Reward: {reward:.3f}")

    # ---------- 保存模型 ----------
    print("\n" + "=" * 60)
    print("训练完成！")
    print("=" * 60)

    save_path = "./ppo_finetuned_model"
    model.save_pretrained(save_path)
    tokenizer.save_pretrained(save_path)
    print(f"\n模型已保存到: {save_path}")


if __name__ == "__main__":
    main()
