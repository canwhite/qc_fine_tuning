"""
RLVR (Reinforcement Learning with Verifiable Rewards) 完全手动实现
================================================================
这是一个透明的 RLVR 实现，每个核心逻辑都手写，让你清楚看到每一步的计算。

核心组件：
1. PolicyModel: 纯策略网络（不需要ValueHead，因为奖励是终态的）
2. ReferenceModel: 冻结的参考模型（用于KL约束）
3. verify_reward: 可验证奖励函数（用户自定义）
4. compute_advantages: 基于排名的优势计算
5. rlvr_update_step: RLVR更新循环

RLVR vs PPO:
- PPO: 需要reward model + value network + GAE
- RLVR: 奖励可直接验证（0/1），不需要value network

运行: python rlvr_manual.py
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from typing import List, Callable
import numpy as np


# ============================================================
# 第一部分：策略模型（不需要ValueHead）
# ============================================================

class PolicyModel(nn.Module):
    """
    策略网络（纯策略，无ValueHead）

    RLVR的核心假设：奖励是终态可验证的，不需要估计状态价值
    """

    def __init__(self, base_model_name: str):
        super().__init__()

        print(f"加载基础模型: {base_model_name}")
        self.base_model = AutoModelForCausalLM.from_pretrained(
            base_model_name,
            torch_dtype=torch.float16,
            trust_remote_code=True
        )

        hidden_size = self.base_model.config.hidden_size
        print(f"✓ 模型加载完成，隐藏层大小: {hidden_size}")

    def forward(self, input_ids, attention_mask=None):
        """
        前向传播

        返回:
            logits: [batch, seq_len, vocab_size] - 策略的 logits
        """
        outputs = self.base_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_dict=True
        )
        logits = outputs.logits  # [batch, seq_len, vocab_size]
        return logits

    def generate(self, input_ids, **kwargs):
        """生成文本"""
        return self.base_model.generate(input_ids, **kwargs)


# ============================================================
# 第二部分：参考模型（冻结）
# ============================================================

class ReferenceModel:
    """
    参考模型：冻结的初始策略

    用于计算 KL 散度，防止策略偏离太远
    """

    def __init__(self, policy_model: PolicyModel):
        print("创建参考模型（冻结参数）")

        self.model = type(policy_model)(policy_model.base_model.name_or_path)
        self.model.load_state_dict(policy_model.state_dict())

        for param in self.model.parameters():
            param.requires_grad = False

        self.model.eval()
        print("✓ 参考模型已冻结")

    @torch.no_grad()
    def get_log_probs(self, input_ids, attention_mask=None):
        """获取参考模型的 log 概率"""
        logits = self.model(input_ids, attention_mask)
        return F.log_softmax(logits, dim=-1)


# ============================================================
# 第三部分：Log Probabilities 计算
# ============================================================

def compute_log_probs(logits: torch.Tensor, input_ids: torch.Tensor) -> torch.Tensor:
    """
    计算 log probabilities

    Args:
        logits: [batch, seq_len, vocab_size]
        input_ids: [batch, seq_len]

    Returns:
        log_probs: [batch, seq_len]
    """
    log_probs = F.log_softmax(logits, dim=-1)
    batch_size, seq_len = input_ids.shape
    log_probs = log_probs.gather(2, input_ids.unsqueeze(2)).squeeze(2)
    return log_probs


# ============================================================
# 第四部分：优势计算（基于排名）
# ============================================================

def compute_advantages_group_relative(
    rewards: torch.Tensor,
    num_samples: int
) -> torch.Tensor:
    """
    GRPO 风格的优势计算：基于排名的相对优势

    对于同一个prompt采样的多个response：
    - 奖励最高的response优势为正
    - 奖励最低的response优势为负
    - 中间的按比例缩放

    Args:
        rewards: [batch_size * num_samples] - 所有采样response的奖励
        num_samples: 每个prompt采样的数量

    Returns:
        advantages: [batch_size * num_samples] - 优势值
    """
    batch_size = rewards.shape[0] // num_samples
    advantages = torch.zeros_like(rewards)

    for i in range(batch_size):
        start = i * num_samples
        end = start + num_samples
        group_rewards = rewards[start:end]

        # 计算相对排名（rank / num_samples - 0.5）
        # 这样最高的优势为正，最低的为负
        sorted_indices = torch.argsort(group_rewards)
        ranks = torch.argsort(sorted_indices).float()  # 0=最低, num_samples-1=最高

        # 归一化到 [-0.5, 0.5]，再乘2得到 [-1, 1]
        normalized_ranks = (ranks / (num_samples - 1)) - 0.5
        advantages[start:end] = normalized_ranks

    return advantages


# ============================================================
# 第五部分：策略损失（简化版REINFORCE）
# ============================================================

def compute_policy_loss(
    log_probs: torch.Tensor,
    advantages: torch.Tensor,
    response_mask: torch.Tensor
) -> torch.Tensor:
    """
    计算策略损失（REINFORCE风格）

    L = -E[log π(a|s) * advantage]

    只在response部分计算损失（prompt部分mask掉）

    Args:
        log_probs: [batch, seq_len] - log概率
        advantages: [batch * num_samples] - 优势（每个prompt的多个sample）
        response_mask: [batch, seq_len] - 标记response位置

    Returns:
        policy_loss: 标量
    """
    # 取response部分的log_probs
    response_log_probs = log_probs * response_mask

    # 平均log_prob（只在response位置）
    seq_len = log_probs.shape[1]
    response_mask_sum = response_mask.sum(dim=1) + 1e-8
    mean_log_probs = response_log_probs.sum(dim=1) / response_mask_sum

    # mean_log_probs 和 advantages 都是 [batch_size * num_samples]，直接对应
    policy_loss = -(mean_log_probs * advantages).mean()

    return policy_loss


# ============================================================
# 第六部分：KL惩罚
# ============================================================

def compute_kl_penalty(
    logits: torch.Tensor,
    ref_log_probs: torch.Tensor,
    input_ids: torch.Tensor,
    response_mask: torch.Tensor
) -> torch.Tensor:
    """
    计算 KL 散度惩罚

    KL(π || π_ref) ≈ Σ π(a|s) * (log π - log π_ref)
    这里近似为: Σ (log π_new[a] - log π_ref[a])，只在 response 位置计算

    Args:
        logits: [batch, seq_len, vocab_size] - 新策略的 logits
        ref_log_probs: [batch, seq_len, vocab_size] - 参考策略的 log prob
        input_ids: [batch, seq_len] - token IDs，用于 gather
        response_mask: [batch, seq_len] - response位置

    Returns:
        kl_div: 标量
    """
    new_log_probs = F.log_softmax(logits, dim=-1)

    # gather：每个位置取实际 token 的 log prob
    # new_log_probs: [batch, seq_len, vocab_size] → [batch, seq_len]
    new_log_probs_gathered = new_log_probs.gather(2, input_ids.unsqueeze(2)).squeeze(2)
    # ref_log_probs: [batch, seq_len, vocab_size] → [batch, seq_len]
    ref_log_probs_gathered = ref_log_probs.gather(2, input_ids.unsqueeze(2)).squeeze(2)

    # 每个 token 的 KL 近似
    token_kl = new_log_probs_gathered - ref_log_probs_gathered  # [batch, seq_len]

    # 只在 response 位置计算
    masked_kl = token_kl * response_mask
    return masked_kl.sum(dim=1).mean()


# ============================================================
# 第七部分：RLVR更新步骤
# ============================================================

def rlvr_update_step(
    model: PolicyModel,
    ref_model: ReferenceModel,
    optimizer: torch.optim.Optimizer,
    batch: dict,
    kl_coef: float = 0.1
) -> dict:
    """
    一步 RLVR 更新

    流程：
    1. 前向传播获取log_probs
    2. 获取参考模型的log_probs
    3. 计算策略损失
    4. 计算KL惩罚
    5. 总损失 = 策略损失 + KL惩罚
    6. 反向传播更新
    """
    input_ids = batch['input_ids']
    attention_mask = batch['attention_mask']
    response_mask = batch['response_mask']
    advantages = batch['advantages']

    # 前向传播
    logits = model(input_ids, attention_mask)
    log_probs = compute_log_probs(logits, input_ids)

    # 参考模型（get_log_probs 已返回 log_softmax，无需重复计算）
    ref_log_probs = ref_model.get_log_probs(input_ids, attention_mask)

    # 计算损失
    policy_loss = compute_policy_loss(log_probs, advantages, response_mask)
    kl_penalty = compute_kl_penalty(logits, ref_log_probs, input_ids, response_mask)

    total_loss = policy_loss + kl_coef * kl_penalty

    # 反向传播
    optimizer.zero_grad()
    total_loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
    optimizer.step()

    return {
        'policy_loss': policy_loss.item(),
        'kl_penalty': kl_penalty.item(),
        'total_loss': total_loss.item(),
        'mean_advantage': advantages.mean().item(),
    }


# ============================================================
# 第八部分：经验收集
# ============================================================

def collect_rollouts(
    model: PolicyModel,
    ref_model: ReferenceModel,
    tokenizer,
    prompts: List[str],
    verify_reward: Callable[[str, str], float],
    device: str,
    num_samples: int = 4,
    max_new_tokens: int = 64
) -> dict:
    """
    收集经验数据

    对每个prompt采样多个response，计算验证奖励
    """
    model.eval()

    all_input_ids = []
    all_attention_mask = []
    all_response_mask = []
    all_log_probs = []
    all_rewards = []

    with torch.no_grad():
        for prompt in prompts:
            # 编码prompt
            prompt_inputs = tokenizer(
                prompt,
                return_tensors="pt",
                padding=False,
                truncation=True,
                max_length=256
            )
            prompt_input_ids = prompt_inputs['input_ids'].to(device)
            prompt_attention_mask = prompt_inputs['attention_mask'].to(device)
            prompt_len = prompt_input_ids.shape[1]

            # 采样多个response
            for _ in range(num_samples):
                generation_output = model.generate(
                    prompt_input_ids,
                    attention_mask=prompt_attention_mask,
                    max_new_tokens=max_new_tokens,
                    do_sample=True,
                    temperature=0.8,
                    top_p=0.9,
                    pad_token_id=tokenizer.pad_token_id,
                )

                # 提取response部分
                response_ids = generation_output[:, prompt_len:]
                response_mask = torch.ones_like(response_ids)

                # 完整序列
                full_input_ids = torch.cat([prompt_input_ids, response_ids], dim=1)
                full_attention_mask = torch.cat([
                    prompt_attention_mask,
                    torch.ones_like(response_ids)
                ], dim=1)

                # 解码并验证奖励
                response_text = tokenizer.decode(response_ids[0], skip_special_tokens=True)
                reward = verify_reward(prompt, response_text)

                # 计算log_probs
                logits = model(full_input_ids, full_attention_mask)
                log_probs = compute_log_probs(logits, full_input_ids)

                all_input_ids.append(full_input_ids)
                all_attention_mask.append(full_attention_mask)
                all_response_mask.append(response_mask)
                all_log_probs.append(log_probs)
                all_rewards.append(reward)

    # Padding到相同长度
    max_len = max(x.shape[1] for x in all_input_ids)

    padded_input_ids = torch.cat([
        F.pad(x, (0, max_len - x.shape[1])) for x in all_input_ids
    ], dim=0)

    padded_attention_mask = torch.cat([
        F.pad(x, (0, max_len - x.shape[1])) for x in all_attention_mask
    ], dim=0)

    padded_response_mask = torch.cat([
        F.pad(x, (0, max_len - x.shape[1])) for x in all_response_mask
    ], dim=0)

    padded_log_probs = torch.cat([
        F.pad(x, (0, max_len - x.shape[1])) for x in all_log_probs
    ], dim=0)

    rewards = torch.tensor(all_rewards, device=device, dtype=torch.float32)

    # 计算优势（GRPO风格）
    advantages = compute_advantages_group_relative(rewards, num_samples)

    return {
        'input_ids': padded_input_ids,
        'attention_mask': padded_attention_mask,
        'response_mask': padded_response_mask,
        'log_probs': padded_log_probs,
        'rewards': rewards,
        'advantages': advantages,
        'prompts': prompts,
    }


# ============================================================
# 第九部分：示例验证奖励函数
# ============================================================

def math_verify_reward(prompt: str, response: str) -> float:
    """
    数学题验证奖励函数

    规则：
    - 如果response包含最终答案数字，且prompt有对应问题 -> 1.0
    - 否则 -> 0.0
    """
    # 简单实现：检查是否包含数字
    has_numbers = any(c.isdigit() for c in response)
    reasonable_length = len(response) > 10
    return 1.0 if (has_numbers and reasonable_length) else 0.0


def length_verify_reward(prompt: str, response: str) -> float:
    """
    长度验证奖励函数（用于测试）

    奖励与response长度成正比
    """
    # 归一化到0-1
    return min(len(response) / 200, 1.0)


# ============================================================
# 第十部分：训练数据
# ============================================================

class RLVRDataset(Dataset):
    """RLVR数据集"""

    def __init__(self, prompts: List[str]):
        self.prompts = prompts

    def __len__(self):
        return len(self.prompts)

    def __getitem__(self, idx):
        return {"prompt": self.prompts[idx]}


TRAINING_PROMPTS = [
    "Solve for x: 2x + 5 = 15. What is x?",
    "What is the derivative of x^2?",
    "Calculate: 15 + 27 = ?",
    "What is 20% of 150?",
    "Simplify: 3(x + 2) - 2x",
    "What is the square root of 144?",
    "Solve: y = 2x + 3, when x = 4, what is y?",
    "What is 25 * 4?",
    "Calculate: 100 - 37",
    "What is 3^3 (3 cubed)?",
]


# ============================================================
# 第十一部分：主训练流程
# ============================================================

def train_rlvr():
    """RLVR训练主函数"""

    print("=" * 70)
    print("RLVR (Reinforcement Learning with Verifiable Rewards) 手动实现")
    print("=" * 70)

    # 配置
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"\n设备: {device}")

    model_name = "Qwen/Qwen2.5-0.5B-Instruct"
    num_samples = 4  # 每个prompt采样的数量
    num_epochs = 3
    batch_size = 2
    learning_rate = 1e-5
    kl_coef = 0.1

    # ========== 1. 加载Tokenizer ==========
    print("\n[1/5] 加载Tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    print(f"✓ 词表大小: {len(tokenizer)}")

    # ========== 2. 创建模型 ==========
    print("\n[2/5] 创建策略模型...")
    model = PolicyModel(model_name).to(device)

    # ========== 3. 创建参考模型 ==========
    print("\n[3/5] 创建参考模型...")
    ref_model = ReferenceModel(model)

    # ========== 4. 创建优化器 ==========
    print("\n[4/5] 创建优化器...")
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    print(f"✓ 学习率: {learning_rate}")

    # ========== 5. 准备数据 ==========
    print("\n[5/5] 准备训练数据...")
    dataset = RLVRDataset(TRAINING_PROMPTS)
    verify_reward = math_verify_reward  # 可替换为自定义验证函数
    print(f"✓ 数据集大小: {len(dataset)}")
    print(f"✓ 每prompt采样数: {num_samples}")

    # ========== 训练循环 ==========
    print("\n" + "=" * 70)
    print("开始训练")
    print("=" * 70)

    for epoch in range(num_epochs):
        print(f"\n{'=' * 70}")
        print(f"Epoch {epoch + 1}/{num_epochs}")
        print(f"{'=' * 70}")

        num_batches = len(dataset) // batch_size
        total_reward = 0
        total_samples = 0

        for batch_idx in range(num_batches):
            # 获取一批prompts
            start_idx = batch_idx * batch_size
            end_idx = start_idx + batch_size
            batch_prompts = [dataset[i]['prompt'] for i in range(start_idx, end_idx)]

            # 收集经验
            print(f"\n[Batch {batch_idx + 1}/{num_batches}] 采样responses...")
            batch_data = collect_rollouts(
                model, ref_model, tokenizer, batch_prompts,
                verify_reward, device, num_samples
            )

            # 统计
            mean_reward = batch_data['rewards'].mean().item()
            total_reward += batch_data['rewards'].sum().item()
            total_samples += len(batch_data['rewards'])

            print(f"  平均验证奖励: {mean_reward:.3f}")

            # RLVR更新
            train_batch = {
                'input_ids': batch_data['input_ids'].to(device),
                'attention_mask': batch_data['attention_mask'].to(device),
                'response_mask': batch_data['response_mask'].to(device),
                'advantages': batch_data['advantages'].to(device),
            }

            stats = rlvr_update_step(model, ref_model, optimizer, train_batch, kl_coef)

            # 每次更新后同步参考模型，保持 KL 约束有效
            ref_model.model.load_state_dict(model.state_dict())

            print(f"  策略损失: {stats['policy_loss']:.4f}")
            print(f"  KL惩罚: {stats['kl_penalty']:.4f}")
            print(f"  总损失: {stats['total_loss']:.4f}")

        # Epoch统计
        print(f"\nEpoch {epoch + 1} 完成:")
        print(f"  全局平均奖励: {total_reward / total_samples:.3f}")

    # ========== 测试 ==========
    print("\n" + "=" * 70)
    print("测试训练后的模型")
    print("=" * 70)

    model.eval()
    test_prompts = [
        "Solve for x: 3x + 6 = 21. What is x?",
        "What is the derivative of 2x^2?",
    ]

    for prompt in test_prompts:
        print(f"\nQ: {prompt}")

        inputs = tokenizer(prompt, return_tensors="pt").to(device)
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
        response = response[len(prompt):].strip()
        print(f"A: {response}")

        reward = verify_reward(prompt, response)
        print(f"验证奖励: {reward:.3f}")

    print("\n" + "=" * 70)
    print("训练完成！")
    print("=" * 70)


if __name__ == "__main__":
    train_rlvr()
