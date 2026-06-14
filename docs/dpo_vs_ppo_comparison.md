# DPO vs PPO 完整对比详解

本文档详细对比 DPO (Direct Preference Optimization) 和 PPO (Proximal Policy Optimization) 的训练流程、损失计算和迭代更新过程。

---

## 目录

1. [DPO 损失计算详解](#dpo-损失计算详解)
2. [DPO 迭代更新流程](#dpo-迭代更新流程)
3. [PPO vs DPO 完整对比](#ppo-vs-dpo-完整对比)
4. [核心区别总结](#核心区别总结)

---

## DPO 损失计算详解

### 1. DPO 损失函数

**数学公式**：

```
L_DPO = -log σ(β × ((log π_chosen - log π_ref_chosen)
                     - (log π_rejected - log π_ref_rejected)))
```

**简化理解**：

```
L_DPO = -log sigmoid(β × (chosen_logratio - rejected_logratio))

其中：
- chosen_logratio = log π_chosen - log π_ref_chosen
- rejected_logratio = log π_rejected - log π_ref_rejected
- β: KL 惩罚系数
```

---

### 2. 完整的计算流程

```python
def dpo_loss_computation():
    """完整的 DPO 损失计算"""

    # ========== 步骤 1: 准备数据 ==========
    # 输入：三元组 (prompt, chosen, rejected)
    prompt = "什么是机器学习？"
    chosen = "机器学习是人工智能的一个分支..."
    rejected = "就是学习。"

    # ========== 步骤 2: 编码 ==========
    chosen_text = f"{prompt}{chosen}"
    rejected_text = f"{prompt}{rejected}"

    chosen_ids = tokenizer(chosen_text)
    rejected_ids = tokenizer(rejected_text)

    # ========== 步骤 3: 前向传播 ==========
    # 策略模型
    policy_logits_chosen = policy_model(chosen_ids)
    policy_logits_rejected = policy_model(rejected_ids)

    # 参考模型（冻结，不需要梯度）
    with torch.no_grad():
        ref_logits_chosen = ref_model(chosen_ids)
        ref_logits_rejected = ref_model(rejected_ids)

    # ========== 步骤 4: 计算 log probabilities ==========
    # 策略模型的 log_probs
    policy_log_probs_chosen = compute_log_probs(policy_logits_chosen, chosen_ids)
    policy_log_probs_rejected = compute_log_probs(policy_logits_rejected, rejected_ids)

    # 参考模型的 log_probs
    ref_log_probs_chosen = compute_log_probs(ref_logits_chosen, chosen_ids)
    ref_log_probs_rejected = compute_log_probs(ref_logits_rejected, rejected_ids)

    # ========== 步骤 5: 平均序列长度 ==========
    # 简化：对整个序列求平均
    policy_log_prob_chosen = policy_log_probs_chosen.mean()
    policy_log_prob_rejected = policy_log_probs_rejected.mean()
    ref_log_prob_chosen = ref_log_probs_chosen.mean()
    ref_log_prob_rejected = ref_log_probs_rejected.mean()

    # ========== 步骤 6: 计算 logratios ==========
    chosen_logratio = policy_log_prob_chosen - ref_log_prob_chosen
    rejected_logratio = policy_log_prob_rejected - ref_log_prob_rejected

    # ========== 步骤 7: 计算 DPO 损失 ==========
    beta = 0.1
    logratios_diff = chosen_logratio - rejected_logratio

    # 方法 1: 直接计算
    sigmoid_logits = torch.sigmoid(beta * logratios_diff)
    loss = -torch.log(sigmoid_logits)

    # 方法 2: 数值稳定版本（推荐）
    loss = -F.logsigmoid(beta * logratios_diff)

    return loss
```

---

### 3. 数值示例

```python
# ========== 假设数据 ==========
# 策略模型的 log 概率（平均）
policy_log_prob_chosen = -2.5     # 策略对 chosen 的 log 概率
policy_log_prob_rejected = -4.0   # 策略对 rejected 的 log 概率

# 参考模型的 log 概率（平均）
ref_log_prob_chosen = -2.8        # 参考对 chosen 的 log 概率
ref_log_prob_rejected = -3.5      # 参考对 rejected 的 log 概率

# ========== 计算 logratios ==========
chosen_logratio = policy_log_prob_chosen - ref_log_prob_chosen
               = -2.5 - (-2.8)
               = 0.3

rejected_logratio = policy_log_prob_rejected - ref_log_prob_rejected
                 = -4.0 - (-3.5)
                 = -0.5

# ========== 计算差值 ==========
logratios_diff = chosen_logratio - rejected_logratio
              = 0.3 - (-0.5)
              = 0.8

# ========== 计算 DPO 损失 ==========
beta = 0.1
sigmoid_input = beta * logratios_diff = 0.1 * 0.8 = 0.08
sigmoid_output = sigmoid(0.08) ≈ 0.52
loss = -log(0.52) ≈ -0.6539

# 结果：
# loss ≈ -0.65
# 负数，说明模型做得不错（chosen 概率 > rejected 概率）
```

---

## DPO 迭代更新流程

### 完整代码

```python
def dpo_update_step(
    policy_model: nn.Module,
    ref_model: nn.Module,
    optimizer: torch.optim.Optimizer,
    batch: Dict[str, torch.Tensor],
    beta: float = 0.1
) -> Dict[str, float]:
    """
    DPO 更新步骤

    流程：
    1. 前向传播（策略 + 参考）
    2. 计算 log_probs
    3. 计算 logratios
    4. 计算 DPO 损失
    5. 反向传播
    6. 更新策略模型
    """

    # ========== 1. 前向传播 ==========
    chosen_inputs = {
        'input_ids': batch['chosen_input_ids'],
        'attention_mask': batch['chosen_attention_mask']
    }
    rejected_inputs = {
        'input_ids': batch['rejected_input_ids'],
        'attention_mask': batch['rejected_attention_mask']
    }

    # 策略模型
    policy_logits_chosen = policy_model(**chosen_inputs).logits
    policy_logits_rejected = policy_model(**rejected_inputs).logits

    # ========== 2. 计算 log_probs ==========
    policy_log_probs_chosen = compute_log_probs(policy_logits_chosen, chosen_inputs['input_ids'])
    policy_log_probs_rejected = compute_log_probs(policy_logits_rejected, rejected_inputs['input_ids'])

    # ========== 3. 参考模型（不需要梯度） ==========
    with torch.no_grad():
        ref_logits_chosen = ref_model(**chosen_inputs).logits
        ref_logits_rejected = ref_model(**rejected_inputs).logits

    ref_log_probs_chosen = compute_log_probs(ref_logits_chosen, chosen_inputs['input_ids'])
    ref_log_probs_rejected = compute_log_probs(ref_logits_rejected, rejected_inputs['input_ids'])

    # ========== 4. 平均 ==========
    policy_log_prob_chosen = policy_log_probs_chosen.mean(dim=1)
    policy_log_prob_rejected = policy_log_probs_rejected.mean(dim=1)
    ref_log_prob_chosen = ref_log_probs_chosen.mean(dim=1)
    ref_log_prob_rejected = ref_log_probs_rejected.mean(dim=1)

    # ========== 5. 计算 logratios ==========
    chosen_logratios = policy_log_prob_chosen - ref_log_prob_chosen
    rejected_logratios = policy_log_prob_rejected - ref_log_prob_rejected

    # ========== 6. 计算 DPO 损失 ==========
    logratios_diff = chosen_logratios - rejected_logratios
    loss = -F.logsigmoid(beta * logratios_diff).mean()

    # ========== 7. 反向传播 ==========
    optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(policy_model.parameters(), max_norm=1.0)
    optimizer.step()

    # ========== 8. 统计信息 ==========
    with torch.no_grad():
        accuracy = (logratios_diff > 0).float().mean()

    return {
        'loss': loss.item(),
        'accuracy': accuracy.item(),
        'logratios_diff': logratios_diff.mean().item()
    }
```

---

### DPO 训练循环

```python
# DPO 训练流程
for epoch in range(num_epochs):
    for batch in dataloader:
        # 1. 前向传播（chosen + rejected）
        policy_log_probs_chosen = policy_model(chosen)
        policy_log_probs_rejected = policy_model(rejected)

        # 2. 参考模型（不需要梯度）
        with torch.no_grad():
            ref_log_probs_chosen = ref_model(chosen)
            ref_log_probs_rejected = ref_model(rejected)

        # 3. 计算 logratios
        chosen_logratios = policy_log_probs_chosen - ref_log_probs_chosen
        rejected_logratios = policy_log_probs_rejected - ref_log_probs_rejected

        # 4. 计算 DPO 损失（一项！）
        loss = -F.logsigmoid(beta * (chosen_logratios - rejected_logratios))

        # 5. 反向传播
        loss.backward()
        optimizer.step()
```

---

## PPO vs DPO 完整对比

### 1. 数据需求

```python
# ========== PPO 数据 ==========
PPO_data = {
    'prompt': "什么是机器学习？"
}
# 回答由模型实时生成

# ========== DPO 数据 ==========
DPO_data = {
    'prompt': "什么是机器学习？",
    'chosen': "机器学习是人工智能的一个分支...",
    'rejected': "就是学习。"
}
# 回答已经给定了
```

---

### 2. 训练流程对比

```python
# ========== PPO 训练流程 ==========
for epoch in range(num_epochs):
    for batch in dataloader:
        # 1. 生成回答
        responses = policy_model.generate(prompts)

        # 2. 计算奖励（需要 RM）
        rewards = reward_model(prompts, responses)

        # 3. 计算 log_probs（策略 + 参考）
        policy_log_probs = policy_model(prompts + responses)
        ref_log_probs = ref_model(prompts + responses)

        # 4. 计算 KL 惩罚
        kl_div = policy_log_probs - ref_log_probs

        # 5. 构造奖励
        rewards_with_kl = rewards - kl_coef * kl_div

        # 6. 计算 GAE（需要价值网络）
        values = value_network(prompts + responses)
        advantages, returns = compute_gae(rewards_with_kl, values)

        # 7. 计算损失
        policy_loss = compute_policy_loss(policy_log_probs, old_log_probs, advantages)
        value_loss = compute_value_loss(values, returns)
        total_loss = policy_loss + vf_coef * value_loss

        # 8. 反向传播
        total_loss.backward()
        optimizer.step()

# ========== DPO 训练流程 ==========
for epoch in range(num_epochs):
    for batch in dataloader:
        # 1. 前向传播（chosen + rejected）
        policy_log_probs_chosen = policy_model(chosen)
        policy_log_probs_rejected = policy_model(rejected)

        # 2. 参考模型（不需要梯度）
        with torch.no_grad():
            ref_log_probs_chosen = ref_model(chosen)
            ref_log_probs_rejected = ref_model(rejected)

        # 3. 计算 logratios
        chosen_logratios = policy_log_probs_chosen - ref_log_probs_chosen
        rejected_logratios = policy_log_probs_rejected - ref_log_probs_rejected

        # 4. 计算 DPO 损失（一项！）
        loss = -F.logsigmoid(beta * (chosen_logratios - rejected_logratios))

        # 5. 反向传播
        loss.backward()
        optimizer.step()
```

---

### 3. 损失函数对比

```python
# ========== PPO 损失 ==========
# 复杂的多项损失
total_loss = policy_loss + value_loss + kl_penalty

# 策略损失（PPO 裁剪）
ratio = torch.exp(log_probs - old_log_probs)
clipped_ratio = torch.clamp(ratio, 0.8, 1.2)
policy_loss = -torch.min(advantages * ratio, advantages * clipped_ratio)

# 价值损失（MSE）
value_loss = 0.5 * (values - returns) ** 2

# KL 惩罚
kl_penalty = kl_coef * (ref_log_probs - log_probs)

# ========== DPO 损失 ==========
# 简单的单项损失
loss = -F.logsigmoid(beta * (chosen_logratios - rejected_logratios))

# 展开：
loss = -log σ(β × ((log π_chosen - log π_ref_chosen)
                   - (log π_rejected - log π_ref_rejected)))
```

---

### 4. 模型组件对比

```python
# ========== PPO 需要的组件 ==========
PPO_components = {
    'policy_model': '生成文本',
    'value_head': '预测状态价值',
    'ref_model': '计算 KL 惩罚',
    'reward_model': '评估生成质量'
}

# ========== DPO 需要的组件 ==========
DPO_components = {
    'policy_model': '生成文本',
    'ref_model': '计算 logratios'
    # 不需要价值头和奖励模型！
}
```

---

### 5. 完整对比表

| 维度 | PPO | DPO |
|------|-----|-----|
| **训练数据** | 只有 prompt | (prompt, chosen, rejected) |
| **训练方式** | 在线训练 (on-policy) | 离线训练 (off-policy) |
| **模型架构** | Policy + ValueHead + Reference + Reward Model | Policy + Reference |
| **奖励信号** | 需要训练好的 Reward Model | 不需要显式奖励模型 |
| **损失函数** | L_CLIP + L_VF + KL_penalty | -log sigmoid(β × Δ) |
| **训练稳定性** | 较复杂，需要调节多个超参数 | 更稳定，超参数较少 |
| **计算成本** | 较高（需要在线生成 + 训练多个模型） | 较低（离线训练，只需要一个模型） |
| **是否需要生成** | ✅ 每次都要生成 | ❌ 不需要生成 |
| **是否需要 RM** | ✅ 需要 | ❌ 不需要 |
| **是否需要 ValueHead** | ✅ 需要 | ❌ 不需要 |
| **更新频率** | 生成一次，多次更新 | 直接更新 |
| **样本效率** | 较低 | 较高 |

---

## 核心区别总结

### 1. 奖励来源

```python
# PPO: 显式奖励模型
rewards = reward_model(prompt, response)
# 需要：训练好的 RM

# DPO: 隐式偏好
chosen vs rejected
# 不需要：RM，直接从偏好学习
```

---

### 2. 优化目标

```python
# PPO: 最大化累积奖励
max E[R] = max E[Σ γ^t × r_t]

# DPO: 最大化偏好概率
max P(y_chosen > y_rejected | x)
```

---

### 3. 计算复杂度

```python
# PPO:
# - 生成：O(batch_size × seq_len × vocab_size)
# - RM 评估：O(batch_size × seq_len)
# - GAE：O(batch_size × seq_len)
# - 价值网络：O(batch_size × seq_len × hidden_size)
# 总计：复杂度高

# DPO:
# - 前向传播：O(2 × batch_size × seq_len × vocab_size)
#   （chosen + rejected）
# 总计：复杂度低
```

---

### 4. 训练稳定性

```python
# PPO:
# - 需要调节多个超参数
# - 价值网络可能不准确
# - KL 惩罚可能不稳定

# DPO:
# - 超参数少（主要是 beta）
# - 不需要价值网络
# - 更稳定
```

---

### 5. 为什么 DPO 更简单？

**数学原理**：

```python
# PPO 的优化目标
max E[R] - KL_divergence

# 其中 R = reward_model(prompt, response)
# 需要训练 reward_model

# DPO 的优化目标
max P(y_chosen > y_rejected | x)

# 在 Bradley-Terry 模型下
# P(y_chosen > y_rejected) = σ(R_chosen - R_rejected)

# 其中 R_chosen = log π_chosen - log π_ref_chosen
# 不需要 reward_model！
```

---

## 选择建议

```python
# 使用 PPO 当：
# - 需要在线学习
# - 有好的奖励模型
# - 任务需要实时反馈
# - 有足够计算资源

# 使用 DPO 当：
# - 有大量偏好数据
# - 不想训练奖励模型
# - 计算资源有限
# - 需要快速迭代
```

---

## 总结

### PPO 的特点

**优点**：
- 在线学习，可以持续改进
- 理论基础扎实
- 适合需要实时反馈的任务

**缺点**：
- 需要训练奖励模型
- 需要价值网络
- 计算成本高
- 超参数多，调试复杂

---

### DPO 的特点

**优点**：
- 不需要奖励模型
- 不需要价值网络
- 计算成本低
- 超参数少，易于调试
- 直接从偏好学习

**缺点**：
- 需要偏好数据（人工标注）
- 离线学习，无法实时反馈
- 依赖偏好数据的质量

---

### 核心区别

**一句话总结**：

```
PPO: 在线生成 → RM 评估 → 价值网络 → GAE → PPO 更新（复杂）
DPO: 离线偏好 → 前向传播 → logratios → DPO 更新（简单）
```

**本质区别**：

```
PPO: 需要显式奖励信号，通过强化学习优化
DPO: 直接从偏好学习，避免训练奖励模型
```

这就是 DPO 和 PPO 的完整对比！DPO 通过直接优化偏好目标，避免了奖励模型的训练，大大简化了流程并提高了效率。
