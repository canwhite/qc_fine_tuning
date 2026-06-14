"""
PPO-RLHF 完全手动实现
=====================
这是一个透明的 PPO 实现，每个核心逻辑都手写，让你清楚看到每一步的计算。

核心组件：
1. PolicyWithValueHead: 给语言模型手动添加价值头
2. compute_advantages_gae: 手动实现 GAE 优势计算
3. compute_policy_loss: 手动实现 PPO 策略损失
4. compute_value_loss: 手动实现价值损失
5. ppo_update: 手动实现 PPO 更新循环

运行: python ppo_manual.py
"""

from typing import List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

# ============================================================
# 第一部分：给语言模型添加价值头
# ============================================================

class PolicyWithValueHead(nn.Module):
    """
    策略网络 + 价值网络

    架构：
    - Transformer 主干：共享的特征提取器
    - LM Head：输出 next token 的 logits（策略）
    - Value Head：输出状态价值 V(s)
    """

    def __init__(self, base_model_name: str):
        super().__init__()

        # 1. 加载预训练的语言模型（只保留这个用 HG）
        print(f"加载基础模型: {base_model_name}")
        self.base_model = AutoModelForCausalLM.from_pretrained(
            base_model_name,
            torch_dtype=torch.float16,
            trust_remote_code=True
        )

        # 获取隐藏层大小
        hidden_size = self.base_model.config.hidden_size

        # 2. 手动添加价值头
        # 输入：hidden_state
        # 输出：标量 value
        self.value_head = nn.Sequential(
            #低一层，降维
            nn.Linear(hidden_size, hidden_size // 2),
            nn.Tanh(),
            nn.Linear(hidden_size // 2, 1)
        )

        print(f"✓ 模型加载完成")
        print(f"  - 隐藏层大小: {hidden_size}")
        print(f"  - 价值头: Linear({hidden_size} -> {hidden_size//2} -> 1)")

    def forward(self, input_ids, attention_mask=None):
        """
        前向传播

        返回:
            logits: [batch, seq_len, vocab_size] - 策略的 logits
            hidden_states: [batch, seq_len, hidden_size] - 隐藏状态
            values: [batch, seq_len] - 状态价值
        """
        # 获取 Transformer 输出
        outputs = self.base_model.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
            return_dict=True
        )

        # 最后一层的隐藏状态
        hidden_states = outputs.last_hidden_state  # [batch, seq_len, hidden_size]

        # 1. 策略：通过 LM Head 得到 logits
        logits = self.base_model.lm_head(hidden_states)  # [batch, seq_len, vocab_size]

        # 2. 价值：通过价值头得到 V(s)
        values = self.value_head(hidden_states).squeeze(-1)  # [batch, seq_len]

        return logits, hidden_states, values

    def generate(self, input_ids, **kwargs):
        """生成文本（复用 base_model 的 generate）"""
        return self.base_model.generate(input_ids, **kwargs)

    def get_log_probs(self, logits, input_ids):
        """
        计算 log probabilities

        Args:
            logits: [batch, seq_len, vocab_size]
            input_ids: [batch, seq_len]

        Returns:
            log_probs: [batch, seq_len]
        """
        # log_softmax 得到 log 概率
        log_probs = F.log_softmax(logits, dim=-1)  # [batch, seq_len, vocab_size]

        # 收集对应 token 的 log 概率
        # 比如：位置 i 的 token 是 100，就取 log_probs[:, i, 100]
        batch_size, seq_len = input_ids.shape
        log_probs = log_probs.gather(2, input_ids.unsqueeze(2)).squeeze(2)

        return log_probs


# ============================================================
# 第二部分：参考模型（用于计算 KL 惩罚）
# ============================================================

class ReferenceModel:
    """
    参考模型：冻结的初始策略

    用于计算 KL 散度，防止策略偏离太远
    """

    def __init__(self, policy_model: PolicyWithValueHead):
        print("创建参考模型（冻结参数）")

        # 深拷贝策略模型
        self.model = type(policy_model)(policy_model.base_model.name_or_path)
        self.model.load_state_dict(policy_model.state_dict())

        # 冻结所有参数
        for param in self.model.parameters():
            param.requires_grad = False

        self.model.eval()
        print("✓ 参考模型已冻结")

    def get_log_probs(self, input_ids, attention_mask=None):
        """获取参考模型的 log 概率"""
        with torch.no_grad():
            logits, _, _ = self.model(input_ids, attention_mask)
            log_probs = F.log_softmax(logits, dim=-1)
            return log_probs


# ============================================================
# 第三部分：GAE 优势计算（手动实现）
# ============================================================

def compute_advantages_gae(
    rewards: torch.Tensor,
    values: torch.Tensor,
    gamma: float = 0.99,
    lambda_gae: float = 0.95
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    手动实现 GAE (Generalized Advantage Estimation)

    数学公式：
        δ_t = r_t + γ×V(s_{t+1}) - V(s_t)           # TD 误差
        A_t = δ_t + γλ×δ_{t+1} + (γλ)²×δ_{t+2} + ... # GAE
        R_t = A_t + V(s_t)                          # 回报

    Args:
        rewards: [batch_size, seq_len] - 每个位置的奖励
        values: [batch_size, seq_len] - 每个位置的价值
        gamma: 折扣因子
        lambda_gae: GAE 参数

    Returns:
        advantages: [batch_size, seq_len] - 优势函数
        returns: [batch_size, seq_len] - 回报
    """
    batch_size, seq_len = rewards.shape
    advantages = torch.zeros_like(rewards)
    returns = torch.zeros_like(rewards)

    # 对每个样本独立计算
    for i in range(batch_size):
        # 从后向前累积（递归形式更高效）
        advantage = 0
        for t in reversed(range(seq_len)):
            # 最后一个位置：没有 next value
            if t == seq_len - 1:
                delta = rewards[i, t] - values[i, t]
            else:
                delta = rewards[i, t] + gamma * values[i, t + 1] - values[i, t]

            # GAE 递归公式：A_t = δ_t + γλ × A_{t+1}
            advantage = delta + gamma * lambda_gae * advantage
            advantages[i, t] = advantage

        # 计算回报：R_t = A_t + V(s_t)
        returns[i] = advantages[i] + values[i]

    return advantages, returns


# ============================================================
# 第四部分：PPO 损失计算（手动实现）
# ============================================================

def compute_policy_loss(
    log_probs: torch.Tensor,
    old_log_probs: torch.Tensor,
    advantages: torch.Tensor,
    clip_ratio: float = 0.2
) -> torch.Tensor:
    """
    手动实现 PPO 策略损失

    数学公式：
        ratio = π_new(a|s) / π_old(a|s) = exp(log π_new - log π_old)
        L_CLIP = -min(A × ratio, A × clip(ratio, 1-ε, 1+ε))

    Args:
        log_probs: [batch_size, seq_len] - 新策略的 log 概率
        old_log_probs: [batch_size, seq_len] - 旧策略的 log 概率
        advantages: [batch_size, seq_len] - 优势函数
        clip_ratio: ε 参数（通常 0.2）

    Returns:
        policy_loss: 标量 - 策略损失
    """
    # 1. 计算概率比
    # ratio = exp(log π_new - log π_old)
    log_ratio = log_probs - old_log_probs
    ratio = torch.exp(log_ratio)

    # 2. 裁剪 ratio
    clipped_ratio = torch.clamp(ratio, 1 - clip_ratio, 1 + clip_ratio)

    # 3. 计算 PPO 损失
    # L = -min(A × ratio, A × clipped_ratio)
    policy_loss = -torch.min(
        advantages * ratio,
        advantages * clipped_ratio
    )

    # 只对生成的部分计算损失（mask 掉 prompt）
    # 这里简化处理，直接平均
    return policy_loss.mean()


def compute_value_loss(
    values: torch.Tensor,
    returns: torch.Tensor
) -> torch.Tensor:
    """
    手动实现价值损失

    数学公式：
        L_VF = 0.5 × (V(s_t) - R_t)²

    Args:
        values: [batch_size, seq_len] - 预测的价值
        returns: [batch_size, seq_len] - 真实的回报

    Returns:
        value_loss: 标量 - 价值损失
    """
    # MSE 损失（0.5 是数学上的便利）
    value_loss = 0.5 * F.mse_loss(values, returns)
    return value_loss


def compute_kl_penalty(
    log_probs: torch.Tensor,
    ref_log_probs: torch.Tensor
) -> torch.Tensor:
    """
    计算 KL 散度惩罚

    数学公式：
        KL(π || π_ref) = Σ π(a|s) × log(π(a|s) / π_ref(a|s))
                      ≈ log π(a|s) - log π_ref(a|s)  # 近似

    Args:
        log_probs: [batch_size, seq_len] - 新策略 log 概率
        ref_log_probs: [batch_size, seq_len, vocab_size] - 参考策略 log 概率

    Returns:
        kl_div: [batch_size, seq_len] - KL 散度
    """
    # 近似 KL：log π_new - log π_ref
    kl_div = log_probs.unsqueeze(2) - ref_log_probs
    kl_div = (kl_div.exp() * kl_div).sum(dim=2)  # Σ p × log(p/q)
    return kl_div


# ============================================================
# 第五部分：PPO 更新循环（手动实现）
# ============================================================

def ppo_update_step(
    model: PolicyWithValueHead,
    ref_model: ReferenceModel,
    optimizer: torch.optim.Optimizer,
    batch: dict,
    clip_ratio: float = 0.2,
    vf_coef: float = 0.5,
    kl_coef: float = 0.1
) -> dict:
    """
    手动实现一步 PPO 更新

    完整流程：
    1. 前向传播获取新策略的 log_probs 和 values
    2. 计算优势（GAE）和回报
    3. 计算 KL 惩罚
    4. 计算策略损失（PPO 裁剪）
    5. 计算价值损失（MSE）
    6. 总损失 = 策略损失 + 价值损失 + KL 惩罚
    7. 反向传播更新

    Args:
        model: 策略 + 价值网络
        ref_model: 参考模型（冻结）
        optimizer: 优化器
        batch: 训练数据 batch
        clip_ratio: PPO 裁剪参数
        vf_coef: 价值损失权重
        kl_coef: KL 惩罚系数

    Returns:
        stats: 训练统计信息
    """
    # ========== 1. 前向传播 ==========
    input_ids = batch['input_ids']
    attention_mask = batch['attention_mask']
    old_log_probs = batch['old_log_probs']
    advantages = batch['advantages']
    returns = batch['returns']

    # 新策略的前向传播
    logits, _, values = model(input_ids, attention_mask)

    # 计算 log_probs
    log_probs = model.get_log_probs(logits, input_ids)

    # ========== 2. 计算 KL 惩罚 ==========
    with torch.no_grad():
        ref_logits, _, _ = ref_model.model(input_ids, attention_mask)
        ref_log_probs = F.log_softmax(ref_logits, dim=-1)

    kl_div = compute_kl_penalty(log_probs, ref_log_probs)
    kl_penalty = kl_coef * kl_div.mean()

    # ========== 3. 计算策略损失 ==========
    # 归一化优势（稳定训练）
    advantages_normalized = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

    policy_loss = compute_policy_loss(
        log_probs,
        old_log_probs,
        advantages_normalized,
        clip_ratio
    )

    # ========== 4. 计算价值损失 ==========
    value_loss = compute_value_loss(values, returns)

    # ========== 5. 总损失 ==========
    # L = L_policy + vf_coef × L_value + KL_penalty
    total_loss = policy_loss + vf_coef * value_loss + kl_penalty

    # ========== 6. 反向传播 ==========
    optimizer.zero_grad()
    total_loss.backward()

    # 梯度裁剪（防止梯度爆炸）
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

    optimizer.step()

    # ========== 7. 统计信息 ==========
    stats = {
        'policy_loss': policy_loss.item(),
        'value_loss': value_loss.item(),
        'total_loss': total_loss.item(),
        'kl_div': kl_div.mean().item(),
        'mean_value': values.mean().item(),
        'mean_advantage': advantages.mean().item(),
    }

    return stats


# ============================================================
# 第六部分：奖励模型（简化版）
# ============================================================

class SimpleRewardModel:
    """
    简化的奖励模型（用于演示）

    实际应该用真实的 Reward Model，这里用规则模拟
    """

    def __call__(self, query: str, response: str) -> float:
        """
        计算奖励分数

        规则：
        - 基础分：0.5
        - 长度奖励：response 越长分越高（最多 0.3）
        - 关键词奖励：包含 "learn", "explain", "example" 等 +0.2
        """
        # 基础分
        reward = 0.5

        # 长度奖励
        length_bonus = min(len(response) / 200, 0.3)
        reward += length_bonus

        # 关键词奖励
        keywords = ['learn', 'explain', 'example', 'important', 'useful']
        for keyword in keywords:
            if keyword in response.lower():
                reward += 0.04

        return min(reward, 1.0)  # 最多 1.0


# ============================================================
# 第七部分：训练数据
# ============================================================

class PPODataset(Dataset):
    """简单的问答数据集"""

    def __init__(self, queries: List[str]):
        self.queries = queries

    def __len__(self):
        return len(self.queries)

    def __getitem__(self, idx):
        return {"query": self.queries[idx]}


TRAINING_QUERIES = [
    "What is machine learning?",
    "Explain deep learning.",
    "How do neural networks learn?",
    "What is a transformer model?",
    "Explain reinforcement learning.",
    "What is supervised learning?",
    "How does backpropagation work?",
    "What is overfitting?",
    "Explain gradient descent.",
    "What is a loss function?",
]


# ============================================================
# 第八部分：主训练流程
# ============================================================

def collect_rollouts(
    model: PolicyWithValueHead,
    ref_model: ReferenceModel,
    tokenizer,
    queries: List[str],
    reward_model,
    device: str,
    max_new_tokens: int = 64
) -> dict:
    """
    收集经验数据

    流程：
    1. 编码 query
    2. 生成 response
    3. 计算 log_probs（旧策略）
    4. 计算 values（旧策略）
    5. 计算奖励
    6. 计算 GAE 优势和回报
    """
    batch_data = {
        'input_ids': [],
        'attention_mask': [],
        'old_log_probs': [],
        'advantages': [],
        'returns': [],
        'queries': [],
        'responses': [],
        'rewards': [],
    }

    model.eval()

    with torch.no_grad():
        for query in queries:
            # 1. 编码 query
            query_inputs = tokenizer(
                query,
                return_tensors="pt",
                padding=False,
                truncation=True,
                max_length=256
            )
            query_input_ids = query_inputs['input_ids'].to(device)
            query_attention_mask = query_inputs['attention_mask'].to(device)

            # 2. 生成 response
            generation_output = model.generate(
                query_input_ids,
                attention_mask=query_attention_mask,
                max_new_tokens=max_new_tokens,
                do_sample=True,
                temperature=0.7,
                top_p=0.9,
                pad_token_id=tokenizer.pad_token_id,
            )

            # 提取生成的部分
            response_ids = generation_output[:, query_input_ids.shape[1]:]

            # 拼接 query + response
            full_input_ids = torch.cat([query_input_ids, response_ids], dim=1)
            full_attention_mask = torch.cat([
                query_attention_mask,
                torch.ones_like(response_ids)
            ], dim=1)

            # 3. 计算旧策略的 log_probs 和 values
            logits, _, values = model(full_input_ids, full_attention_mask)
            log_probs = model.get_log_probs(logits, full_input_ids)

            # 4. 解码 response
            response_text = tokenizer.decode(response_ids[0], skip_special_tokens=True)

            # 5. 计算奖励
            reward = reward_model(query, response_text)

            # 6. 构造奖励序列
            # - 生成位置：-KL 惩罚（每步）
            # - 最后位置：RM_score - KL 惩罚
            seq_len = full_input_ids.shape[1]
            query_len = query_input_ids.shape[1]
            response_len = seq_len - query_len

            # 简化：所有位置给相同的奖励
            rewards = torch.zeros(1, seq_len, device=device)
            rewards[:, query_len:] = reward / response_len  # 平均分配
            rewards[:, -1] += reward * 0.5  # 最后位置额外奖励

            # 7. 计算 GAE
            advantages, returns = compute_advantages_gae(
                rewards,
                values,
                gamma=0.99,
                lambda_gae=0.95
            )

            # 保存数据
            batch_data['input_ids'].append(full_input_ids)
            batch_data['attention_mask'].append(full_attention_mask)
            batch_data['old_log_probs'].append(log_probs)
            batch_data['advantages'].append(advantages)
            batch_data['returns'].append(returns)
            batch_data['queries'].append(query)
            batch_data['responses'].append(response_text)
            batch_data['rewards'].append(reward)

    # Padding 到相同长度
    max_len = max(x.shape[1] for x in batch_data['input_ids'])

    for key in ['input_ids', 'attention_mask', 'old_log_probs', 'advantages', 'returns']:
        batch_data[key] = torch.cat([
            F.pad(x, (0, max_len - x.shape[1])) for x in batch_data[key]
        ], dim=0)

    return batch_data


def train_ppo():
    """PPO 训练主函数"""

    print("=" * 70)
    print("PPO-RLHF 手动实现训练")
    print("=" * 70)

    # 配置
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"\n设备: {device}")

    model_name = "Qwen/Qwen2.5-0.5B-Instruct"
    batch_size = 2
    num_epochs = 3
    ppo_epochs_per_batch = 4  # 每个 batch 更新几次
    learning_rate = 1e-5
    clip_ratio = 0.2
    vf_coef = 0.5
    kl_coef = 0.1

    # ========== 1. 加载 Tokenizer ==========
    print("\n[1/5] 加载 Tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    print(f"✓ 词表大小: {len(tokenizer)}")

    # ========== 2. 创建模型 ==========
    print("\n[2/5] 创建策略模型 + 价值模型...")
    model = PolicyWithValueHead(model_name).to(device)

    # ========== 3. 创建参考模型 ==========
    print("\n[3/5] 创建参考模型...")
    ref_model = ReferenceModel(model)

    # ========== 4. 创建优化器 ==========
    print("\n[4/5] 创建优化器...")
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    print(f"✓ 学习率: {learning_rate}")

    # ========== 5. 创建数据 ==========
    print("\n[5/5] 准备训练数据...")
    dataset = PPODataset(TRAINING_QUERIES)
    reward_model = SimpleRewardModel()
    print(f"✓ 数据集大小: {len(dataset)}")

    # ========== 训练循环 ==========
    print("\n" + "=" * 70)
    print("开始训练")
    print("=" * 70)

    for epoch in range(num_epochs):
        print(f"\n{'=' * 70}")
        print(f"Epoch {epoch + 1}/{num_epochs}")
        print(f"{'=' * 70}")

        # 按 batch_size 处理
        num_batches = len(dataset) // batch_size

        for batch_idx in range(num_batches):
            # 获取一批 queries
            start_idx = batch_idx * batch_size
            end_idx = start_idx + batch_size
            queries = [dataset[i]['query'] for i in range(start_idx, end_idx)]

            # 收集经验
            print(f"\n[Batch {batch_idx + 1}/{num_batches}] 收集经验...")
            batch_data = collect_rollouts(
                model, ref_model, tokenizer, queries, reward_model, device
            )

            print(f"  示例 Query: {queries[0]}")
            print(f"  示例 Response: {batch_data['responses'][0][:60]}...")
            print(f"  示例 Reward: {batch_data['rewards'][0]:.3f}")
            print(f"  平均 Reward: {np.mean(batch_data['rewards']):.3f}")

            # 对同一批数据多次更新（PPO 的核心）
            print(f"  执行 {ppo_epochs_per_batch} 轮 PPO 更新...")

            for ppo_epoch in range(ppo_epochs_per_batch):
                # 构造训练 batch
                train_batch = {
                    'input_ids': batch_data['input_ids'].to(device),
                    'attention_mask': batch_data['attention_mask'].to(device),
                    'old_log_probs': batch_data['old_log_probs'].to(device),
                    'advantages': batch_data['advantages'].to(device),
                    'returns': batch_data['returns'].to(device),
                }

                # PPO 更新
                stats = ppo_update_step(
                    model, ref_model, optimizer, train_batch,
                    clip_ratio, vf_coef, kl_coef
                )

                if ppo_epoch == 0:
                    print(f"    PPO Epoch {ppo_epoch + 1}/{ppo_epochs_per_batch}:")
                    print(f"      策略损失: {stats['policy_loss']:.4f}")
                    print(f"      价值损失: {stats['value_loss']:.4f}")
                    print(f"      总损失: {stats['total_loss']:.4f}")
                    print(f"      KL 散度: {stats['kl_div']:.4f}")
                    print(f"      平均价值: {stats['mean_value']:.4f}")
                    print(f"      平均优势: {stats['mean_advantage']:.4f}")

    # ========== 测试训练后的模型 ==========
    print("\n" + "=" * 70)
    print("测试训练后的模型")
    print("=" * 70)

    model.eval()
    test_queries = [
        "What is deep learning?",
        "How to start learning AI?",
    ]

    for query in test_queries:
        print(f"\nQ: {query}")

        inputs = tokenizer(query, return_tensors="pt").to(device)
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=80,
                do_sample=True,
                temperature=0.8,
                top_p=0.9,
                pad_token_id=tokenizer.pad_token_id,
            )

        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        response = response[len(query):]
        print(f"A: {response}")

        reward = reward_model(query, response)
        print(f"Reward: {reward:.3f}")

    print("\n" + "=" * 70)
    print("训练完成！")
    print("=" * 70)


if __name__ == "__main__":
    train_ppo()
