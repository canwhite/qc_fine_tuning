"""
DPO (Direct Preference Optimization) 完全手动实现
================================================
这是一个透明的 DPO 实现，每个核心逻辑都手写，让你清楚看到每一步的计算。

核心组件：
1. compute_log_probs: 计算 log probabilities
2. compute_dpo_loss: 手动实现 DPO 损失函数
3. dpo_update_step: 手动实现 DPO 更新循环

DPO vs PPO:
- PPO: 需要奖励模型 + 价值网络 + 在线生成
- DPO: 只需要偏好数据 (prompt, chosen, rejected)，不需要奖励模型

运行: python dpo_manual.py
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer
from typing import List, Dict
import numpy as np


# ============================================================
# 第一部分：计算 Log Probabilities
# ============================================================

def compute_log_probs(model, input_ids, attention_mask=None):
    """
    计算模型对输入序列的 log probabilities

    数学公式：
        log π(x) = log_softmax(logits)
        log_prob = Σ log π(x_t | x_<t)  # 每个位置的条件概率之和

    Args:
        model: 语言模型
        input_ids: [batch, seq_len]
        attention_mask: [batch, seq_len]

    Returns:
        log_probs: [batch, seq_len] - 每个位置的 log 概率
    """
    # 前向传播获取 logits
    outputs = model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        return_dict=True
    )
    logits = outputs.logits  # [batch, seq_len, vocab_size]

    # 计算 log_softmax
    log_probs = F.log_softmax(logits, dim=-1)  # [batch, seq_len, vocab_size]

    # 收集每个位置实际 token 的 log 概率
    # 比如：位置 i 的 token 是 100，就取 log_probs[:, i, 100]
    batch_size, seq_len, vocab_size = log_probs.shape

    # 展开维度以便 gather
    input_ids_expanded = input_ids.unsqueeze(-1)  # [batch, seq_len, 1]

    # 收集对应 token 的 log 概率
    log_probs = log_probs.gather(2, input_ids_expanded).squeeze(-1)  # [batch, seq_len]

    return log_probs


# ============================================================
# 第二部分：DPO 损失函数（手动实现）
# ============================================================

def compute_dpo_loss(
    policy_log_probs_chosen: torch.Tensor,
    policy_log_probs_rejected: torch.Tensor,
    ref_log_probs_chosen: torch.Tensor,
    ref_log_probs_rejected: torch.Tensor,
    beta: float = 0.1
) -> torch.Tensor:
    """
    手动实现 DPO 损失函数

    数学公式：
        L_DPO = -log σ(β × (log π_chosen - log π_rejected
                           - log π_ref_chosen + log π_ref_rejected))

        = -log σ(β × ((log π_chosen - log π_ref_chosen)
                      - (log π_rejected - log π_ref_rejected)))

    其中：
        - π_chosen: 策略模型对 chosen 的 log 概率
        - π_rejected: 策略模型对 rejected 的 log 概率
        - π_ref_chosen: 参考模型对 chosen 的 log 概率
        - π_ref_rejected: 参考模型对 rejected 的 log 概率
        - β: KL 惩罚系数

    直观理解：
        - (log π_chosen - log π_ref_chosen): 策略偏离参考的程度（chosen）
        - (log π_rejected - log π_ref_rejected): 策略偏离参考的程度（rejected）
        - 我们希望：chosen 的偏离 > rejected 的偏离
        - 即：chosen 更"不可能"来自参考模型，更"可能"来自策略模型

    Args:
        policy_log_probs_chosen: [batch] - 策略对 chosen 的平均 log 概率
        policy_log_probs_rejected: [batch] - 策略对 rejected 的平均 log 概率
        ref_log_probs_chosen: [batch] - 参考对 chosen 的平均 log 概率
        ref_log_probs_rejected: [batch] - 参考对 rejected 的平均 log 概率
        beta: KL 惩罚系数

    Returns:
        loss: 标量 - DPO 损失
    """
    # 1. 计算 log 概率差（策略 vs 参考）
    # 这衡量了策略偏离参考的程度
    chosen_logratios = policy_log_probs_chosen - ref_log_probs_chosen
    rejected_logratios = policy_log_probs_rejected - ref_log_probs_rejected

    # 2. 计算 DPO 的核心项
    # chosen_logratios - rejected_logratios
    # = (log π_chosen - log π_ref_chosen) - (log π_rejected - log π_ref_rejected)
    # = log π_chosen - log π_rejected - (log π_ref_chosen - log π_ref_rejected)
    # ≈ log π_chosen - log π_rejected  (如果参考模型对两者概率相似)
    logratios_diff = chosen_logratios - rejected_logratios

    # 3. 应用 sigmoid
    # sigmoid(x) = 1 / (1 + exp(-x))
    # 我们希望：chosen 的概率 > rejected 的概率
    # 即：logratios_diff > 0
    # sigmoid(logratios_diff) 接近 1
    # -log(sigmoid(...)) 接近 0（损失小）
    sigmoid_logratios = torch.sigmoid(beta * logratios_diff)

    # 4. 计算 DPO 损失
    # L = -log(sigmoid(β × (chosen_logratios - rejected_logratios)))
    # 使用 log_sigmoid 数值更稳定
    loss = -F.logsigmoid(beta * logratios_diff)

    return loss.mean()


def compute_dpo_loss_with_margin(
    policy_log_probs_chosen: torch.Tensor,
    policy_log_probs_rejected: torch.Tensor,
    ref_log_probs_chosen: torch.Tensor,
    ref_log_probs_rejected: torch.Tensor,
    beta: float = 0.1,
    margin: float = 0.0
) -> torch.Tensor:
    """
    带 Margin 的 DPO 损失（变体）

    数学公式：
        L = -log σ(β × ((chosen_logratios - rejected_logratios) - margin))

    Margin 的作用：
        - 要求 chosen 不仅要优于 rejected
        - 而且要优出一个 margin
        - 可以让训练更严格

    Args:
        margin: 额外的间隔要求
    """
    chosen_logratios = policy_log_probs_chosen - ref_log_probs_chosen
    rejected_logratios = policy_log_probs_rejected - ref_log_probs_rejected

    logratios_diff = chosen_logratios - rejected_logratios

    # 减去 margin
    loss = -F.logsigmoid(beta * (logratios_diff - margin))

    return loss.mean()


# ============================================================
# 第三部分：DPO 训练步骤（手动实现）
# ============================================================

def dpo_update_step(
    policy_model: nn.Module,
    ref_model: nn.Module,
    optimizer: torch.optim.Optimizer,
    batch: Dict[str, torch.Tensor],
    beta: float = 0.1
) -> Dict[str, float]:
    """
    手动实现一步 DPO 更新

    完整流程：
    1. 编码 (prompt, chosen) 和 (prompt, rejected)
    2. 计算策略模型的 log_probs
    3. 计算参考模型的 log_probs（不需要梯度）
    4. 计算 DPO 损失
    5. 反向传播更新策略模型

    Args:
        policy_model: 策略模型（会更新）
        ref_model: 参考模型（冻结）
        optimizer: 优化器
        batch: 包含 chosen_inputs 和 rejected_inputs
        beta: KL 惩罚系数

    Returns:
        stats: 训练统计信息
    """
    # ========== 1. 解包 batch ==========
    chosen_inputs = {
        'input_ids': batch['chosen_input_ids'],
        'attention_mask': batch['chosen_attention_mask']
    }
    rejected_inputs = {
        'input_ids': batch['rejected_input_ids'],
        'attention_mask': batch['rejected_attention_mask']
    }

    # ========== 2. 计算策略模型的 log_probs ==========
    policy_log_probs_chosen = compute_log_probs(
        policy_model,
        **chosen_inputs
    )  # [batch, seq_len]

    policy_log_probs_rejected = compute_log_probs(
        policy_model,
        **rejected_inputs
    )  # [batch, seq_len]

    # ========== 3. 计算参考模型的 log_probs（不需要梯度） ==========
    with torch.no_grad():
        ref_log_probs_chosen = compute_log_probs(
            ref_model,
            **chosen_inputs
        )

        ref_log_probs_rejected = compute_log_probs(
            ref_model,
            **rejected_inputs
        )

    # ========== 4. 平均序列长度（简化处理） ==========
    # 实际应该只对 response 部分求平均
    # 这里为了简化，对整个序列求平均
    policy_log_probs_chosen_mean = policy_log_probs_chosen.mean(dim=1)  # [batch]
    policy_log_probs_rejected_mean = policy_log_probs_rejected.mean(dim=1)  # [batch]
    ref_log_probs_chosen_mean = ref_log_probs_chosen.mean(dim=1)  # [batch]
    ref_log_probs_rejected_mean = ref_log_probs_rejected.mean(dim=1)  # [batch]

    # ========== 5. 计算 DPO 损失 ==========
    loss = compute_dpo_loss(
        policy_log_probs_chosen_mean,
        policy_log_probs_rejected_mean,
        ref_log_probs_chosen_mean,
        ref_log_probs_rejected_mean,
        beta
    )

    # ========== 6. 反向传播 ==========
    optimizer.zero_grad()
    loss.backward()

    # 梯度裁剪（防止梯度爆炸）
    torch.nn.utils.clip_grad_norm_(policy_model.parameters(), max_norm=1.0)

    optimizer.step()

    # ========== 7. 统计信息 ==========
    with torch.no_grad():
        # 计算 chosen 和 rejected 的概率差
        chosen_logratios = policy_log_probs_chosen_mean - ref_log_probs_chosen_mean
        rejected_logratios = policy_log_probs_rejected_mean - ref_log_probs_rejected_mean
        logratios_diff = chosen_logratios - rejected_logratios

        # 计算 accuracy（chosen 概率 > rejected 概率）
        accuracy = (logratios_diff > 0).float().mean()

    stats = {
        'loss': loss.item(),
        'accuracy': accuracy.item(),
        'chosen_logratios_mean': chosen_logratios.mean().item(),
        'rejected_logratios_mean': rejected_logratios.mean().item(),
        'logratios_diff_mean': logratios_diff.mean().item(),
        'policy_log_probs_chosen': policy_log_probs_chosen_mean.mean().item(),
        'policy_log_probs_rejected': policy_log_probs_rejected_mean.mean().item(),
    }

    return stats


# ============================================================
# 第四部分：偏好数据集
# ============================================================

class PreferenceDataset(Dataset):
    """
    DPO 偏好数据集

    数据格式：每个样本是 (prompt, chosen, rejected) 三元组
    """

    def __init__(self, data: List[Dict[str, str]]):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


# 示例偏好数据
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
# 第五部分：数据预处理
# ============================================================

def collate_fn(batch: List[Dict], tokenizer, max_length: int = 512):
    """
    将 batch 数据转换为模型输入格式

    Args:
        batch: List of (prompt, chosen, rejected)
        tokenizer: 分词器
        max_length: 最大序列长度

    Returns:
        processed_batch: 包含 chosen_inputs 和 rejected_inputs
    """
    chosen_texts = [f"{item['prompt']}{item['chosen']}" for item in batch]
    rejected_texts = [f"{item['prompt']}{item['rejected']}" for item in batch]

    # Tokenize chosen
    chosen_inputs = tokenizer(
        chosen_texts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=max_length
    )

    # Tokenize rejected
    rejected_inputs = tokenizer(
        rejected_texts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=max_length
    )

    return {
        'chosen_input_ids': chosen_inputs['input_ids'],
        'chosen_attention_mask': chosen_inputs['attention_mask'],
        'rejected_input_ids': rejected_inputs['input_ids'],
        'rejected_attention_mask': rejected_inputs['attention_mask'],
    }


# ============================================================
# 第六部分：主训练流程
# ============================================================

def train_dpo():
    """DPO 训练主函数"""

    print("=" * 70)
    print("DPO (Direct Preference Optimization) 手动实现训练")
    print("=" * 70)

    # 配置
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"\n设备: {device}")

    model_name = "Qwen/Qwen2.5-0.5B-Instruct"
    batch_size = 2
    num_epochs = 3
    learning_rate = 5e-7  # DPO 通常用较小学习率
    beta = 0.1  # KL 惩罚系数
    max_length = 512

    # ========== 1. 加载 Tokenizer ==========
    print("\n[1/5] 加载 Tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    print(f"✓ 词表大小: {len(tokenizer)}")

    # ========== 2. 加载策略模型 ==========
    print("\n[2/5] 加载策略模型...")
    policy_model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        trust_remote_code=True
    ).to(device)
    print(f"✓ 策略模型已加载")

    # ========== 3. 加载参考模型（冻结） ==========
    print("\n[3/5] 加载参考模型...")
    ref_model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        trust_remote_code=True
    ).to(device)
    ref_model.eval()

    # 冻结参考模型的所有参数
    for param in ref_model.parameters():
        param.requires_grad = False

    print(f"✓ 参考模型已加载并冻结")

    # ========== 4. 创建优化器 ==========
    print("\n[4/5] 创建优化器...")
    optimizer = torch.optim.AdamW(policy_model.parameters(), lr=learning_rate)
    print(f"✓ 学习率: {learning_rate}")
    print(f"✓ Beta (KL 系数): {beta}")

    # ========== 5. 准备数据 ==========
    print("\n[5/5] 准备训练数据...")
    dataset = PreferenceDataset(PREFERENCE_DATA)
    print(f"✓ 数据集大小: {len(dataset)}")

    # ========== 训练循环 ==========
    print("\n" + "=" * 70)
    print("开始训练")
    print("=" * 70)

    # 打印 DPO 公式
    print("\nDPO 损失函数:")
    print("L = -log σ(β × ((log π_chosen - log π_ref_chosen)")
    print("              - (log π_rejected - log π_ref_rejected)))")
    print()
    print("其中:")
    print("  - π_chosen: 策略对 chosen 的概率")
    print("  - π_rejected: 策略对 rejected 的概率")
    print("  - π_ref_chosen: 参考对 chosen 的概率")
    print("  - π_ref_rejected: 参考对 rejected 的概率")
    print("  - β: KL 惩罚系数")
    print()

    num_batches = len(dataset) // batch_size
    global_step = 0

    for epoch in range(num_epochs):
        print(f"\n{'=' * 70}")
        print(f"Epoch {epoch + 1}/{num_epochs}")
        print(f"{'=' * 70}")

        epoch_stats = {
            'loss': [],
            'accuracy': [],
            'logratios_diff': [],
        }

        for batch_idx in range(num_batches):
            # 获取 batch 数据
            start_idx = batch_idx * batch_size
            end_idx = start_idx + batch_size
            batch_data = [dataset[i] for i in range(start_idx, end_idx)]

            # 预处理数据
            batch = collate_fn(batch_data, tokenizer, max_length)
            batch = {k: v.to(device) for k, v in batch.items()}

            # DPO 更新
            stats = dpo_update_step(
                policy_model,
                ref_model,
                optimizer,
                batch,
                beta
            )

            # 记录统计信息
            epoch_stats['loss'].append(stats['loss'])
            epoch_stats['accuracy'].append(stats['accuracy'])
            epoch_stats['logratios_diff'].append(stats['logratios_diff_mean'])

            # 打印进度
            if (batch_idx + 1) % 1 == 0:
                print(f"\n  Batch {batch_idx + 1}/{num_batches}")
                print(f"    损失: {stats['loss']:.4f}")
                print(f"    准确率: {stats['accuracy']:.4f}")
                print(f"    Chosen logratio: {stats['chosen_logratios_mean']:.4f}")
                print(f"    Rejected logratio: {stats['rejected_logratios_mean']:.4f}")
                print(f"    Logratio 差: {stats['logratios_diff_mean']:.4f}")

                # 打印示例
                if batch_idx == 0:
                    print(f"\n    示例 Prompt: {batch_data[0]['prompt']}")
                    print(f"    示例 Chosen: {batch_data[0]['chosen'][:60]}...")
                    print(f"    示例 Rejected: {batch_data[0]['rejected'][:60]}...")

            global_step += 1

        # Epoch 总结
        print(f"\n  {'=' * 50}")
        print(f"  Epoch {epoch + 1} 总结:")
        print(f"  ├─ 平均损失: {np.mean(epoch_stats['loss']):.4f}")
        print(f"  ├─ 平均准确率: {np.mean(epoch_stats['accuracy']):.4f}")
        print(f"  └─ 平均 Logratio 差: {np.mean(epoch_stats['logratios_diff']):.4f}")
        print(f"  {'=' * 50}")

    # ========== 测试训练后的模型 ==========
    print("\n" + "=" * 70)
    print("测试训练后的模型")
    print("=" * 70)

    policy_model.eval()
    test_prompts = [
        "什么是深度学习？",
        "如何提高编程能力？",
    ]

    for prompt in test_prompts:
        print(f"\nQ: {prompt}")

        inputs = tokenizer(prompt, return_tensors="pt").to(device)
        with torch.no_grad():
            outputs = policy_model.generate(
                **inputs,
                max_new_tokens=100,
                do_sample=True,
                temperature=0.7,
                top_p=0.9,
                pad_token_id=tokenizer.pad_token_id,
            )

        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        response = response[len(prompt):]
        print(f"A: {response}")

    print("\n" + "=" * 70)
    print("训练完成！")
    print("=" * 70)


# ============================================================
# 第七部分：DPO vs PPO 详细对比
# ============================================================

def explain_dpo_vs_ppo():
    """打印 DPO vs PPO 的详细对比"""
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
    ║  5. 损失函数                                             ║
    ║     PPO: L_CLIP + L_VF + KL_penalty                      ║
    ║     DPO: L = -log σ(β × (chosen_logratio - rejected_logratio)) ║
    ║                                                          ║
    ║  6. 训练稳定性                                           ║
    ║     PPO: 较复杂，需要调节多个超参数                      ║
    ║     DPO: 更稳定，超参数较少                              ║
    ║                                                          ║
    ║  7. 计算成本                                             ║
    ║     PPO: 较高（需要在线生成 + 训练多个模型）             ║
    ║     DPO: 较低（离线训练，只需要一个模型）                ║
    ║                                                          ║
    ╚══════════════════════════════════════════════════════════╝
    """)


if __name__ == "__main__":
    # 打印对比解释
    explain_dpo_vs_ppo()

    # 运行训练
    train_dpo()
