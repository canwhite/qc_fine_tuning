# PPO 和 DPO 完全手动实现指南

本文档详细介绍了 PPO (Proximal Policy Optimization) 和 DPO (Direct Preference Optimization) 的手动实现，每个核心逻辑都有透明的过程。

---

## 目录

1. [PPO 完全手动实现](#ppo-完全手动实现)
2. [DPO 完全手动实现](#dpo-完全手动实现)
3. [核心概念详解](#核心概念详解)
4. [训练流程对比](#训练流程对比)

---

## PPO 完全手动实现

### 核心组件

#### 1. 策略 + 价值网络

```python
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

        # 加载预训练的语言模型
        self.base_model = AutoModelForCausalLM.from_pretrained(
            base_model_name,
            torch_dtype=torch.float16,
            trust_remote_code=True
        )

        # 获取隐藏层大小
        hidden_size = self.base_model.config.hidden_size

        # 手动添加价值头
        #   Transformer 输出 (768维)
        #     ↓
        # 价值头第一层 (768 → 384)
        #     ↓ 提取价值相关特征
        # Tanh 激活 ([-1, 1])
        #     ↓ 稳定输出
        # 价值头第二层 (384 → 1)
        #     ↓ 综合特征
        # 价值标量 (1维)
        self.value_head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.Tanh(), # -1 1
            nn.Linear(hidden_size // 2, 1)
        )

    def forward(self, input_ids, attention_mask=None):
        """
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
        hidden_states = outputs.last_hidden_state

        # 策略：通过 LM Head 得到 logits
        logits = self.base_model.lm_head(hidden_states)

        # 价值：通过价值头得到 V(s)
        values = self.value_head(hidden_states).squeeze(-1)

        return logits, hidden_states, values
```

---

#### 2. GAE 优势计算

**数学公式**：

```
δ_t = r_t + γ×V(s_{t+1}) - V(s_t)           # TD 误差
A_t = δ_t + γλ×δ_{t+1} + (γλ)²×δ_{t+2} + ... # GAE
R_t = A_t + V(s_t)                          # 回报
```

**手动实现**：

```python
def compute_advantages_gae(
    rewards: torch.Tensor,
    values: torch.Tensor,
    gamma: float = 0.99,
    lambda_gae: float = 0.95
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    手动实现 GAE (Generalized Advantage Estimation)

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
```

---

#### 3. PPO 策略损失

**数学公式**：

```
ratio = π_new(a|s) / π_old(a|s) = exp(log π_new - log π_old)
L_CLIP = -min(A × ratio, A × clip(ratio, 1-ε, 1+ε))
```

**手动实现**：

```python
def compute_policy_loss(
    log_probs: torch.Tensor,
    old_log_probs: torch.Tensor,
    advantages: torch.Tensor,
    clip_ratio: float = 0.2
) -> torch.Tensor:
    """
    手动实现 PPO 策略损失

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
    ).mean()

    return policy_loss
```

---

#### 4. 裁剪机制详解

**裁剪参数的来源**：

```python
clip_ratio = 0.2  # 超参数，通常设为 0.2

# 计算裁剪范围
lower_bound = 1 - clip_ratio  # = 0.8
upper_bound = 1 + clip_ratio  # = 1.2

# 裁剪 ratio
clipped_ratio = torch.clamp(ratio, lower_bound, upper_bound)
```

**Clamp 函数的规则**：

```python
clamped_value = clamp(value, min, max)

# 规则：
if value < min:
    clamped_value = min      # 小于最小值，就用最小值
elif value > max:
    clamped_value = max      # 大于最大值，就用最大值
else:
    clamped_value = value    # 在范围内，保持不变
```

**示例**：

```python
# 示例 1：超过上限
clamp(3.0, 0.8, 1.2) = 1.2  # 3.0 > 1.2，裁剪到 1.2

# 示例 2：低于下限
clamp(0.5, 0.8, 1.2) = 0.8  # 0.5 < 0.8，提升到 0.8

# 示例 3：在范围内
clamp(1.0, 0.8, 1.2) = 1.0  # 在范围内，不变
```

**物理意义**：

```
ratio = π_new / π_old

如果 clip_ratio = 0.2：
- ratio 最多增加到 1.2（概率增加 20%）
- ratio 最多减少到 0.8（概率减少 20%）

防止策略变化太快，保证训练稳定
```

---

#### 5. 价值损失

**数学公式**：

```
L_VF = 0.5 × (V(s_t) - R_t)²
```

**手动实现**：

```python
def compute_value_loss(
    values: torch.Tensor,
    returns: torch.Tensor
) -> torch.Tensor:
    """
    手动实现价值损失

    Args:
        values: [batch_size, seq_len] - 预测的价值
        returns: [batch_size, seq_len] - 真实的回报

    Returns:
        value_loss: 标量 - 价值损失
    """
    # MSE 损失（0.5 是数学上的便利）
    value_loss = 0.5 * F.mse_loss(values, returns)
    return value_loss
```

**价值损失如何回传更新价值网络**：

```python
# 1. 前向传播
logits, hidden_states, values = model(input_ids)
# values = [0.5, 0.6, 0.4, 0.3]  # 价值网络的预测

# 2. 计算回报（真实值）
returns = [0.8, 0.7, 0.6, 0.5]  # 通过奖励计算得出

# 3. 计算损失
value_loss = 0.5 × (values - returns)²
            = 0.5 × ([0.5, 0.6, 0.4, 0.3] - [0.8, 0.7, 0.6, 0.5])²
            = 0.5 × ([-0.3, -0.1, -0.2, -0.2])²
            = 0.045

# 4. 反向传播（自动计算梯度）
optimizer.zero_grad()
value_loss.backward()

# PyTorch 自动计算梯度
# ∂loss/∂values
# ↓
# ∂values/∂value_head（价值头的参数）
# ↓
# ∂value_head/∂hidden_states
# ↓
# ∂hidden_states/∂transformer（共享的参数）

# 5. 更新权重
optimizer.step()  # 用梯度更新权重
```

**梯度的流动路径**：

```
损失 (loss)
    ↓
∂loss/∂values
    ↓
∂values/∂W2（价值头第二层权重）
∂values/∂b2（价值头第二层偏置）
    ↓
∂values/∂hidden_states（继续回传）
    ↓
∂hidden_states/∂W1（价值头第一层权重）
∂hidden_states/∂b1（价值头第一层偏置）
    ↓
∂hidden_states/∂transformer（共享的 Transformer 参数）
```

**为什么梯度更新有效**：

```python
# 场景 1：预测偏低
value_pred = 0.5
value_true = 0.8
∂loss/∂value = 0.5 - 0.8 = -0.3（负数）
W_new = W_old - lr × (-0.3) = W_old + 正数
结果：权重增加 → value_pred 增加

# 场景 2：预测偏高
value_pred = 0.9
value_true = 0.6
∂loss/∂value = 0.9 - 0.6 = +0.3（正数）
W_new = W_old - lr × (+0.3) = W_old - 正数
结果：权重减少 → value_pred 减少
```

---

#### 6. PPO 更新步骤

```python
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
    """
    # 1. 前向传播
    input_ids = batch['input_ids']
    attention_mask = batch['attention_mask']
    old_log_probs = batch['old_log_probs']
    advantages = batch['advantages']
    returns = batch['returns']

    # 新策略的前向传播
    logits, _, values = model(input_ids, attention_mask)
    log_probs = model.get_log_probs(logits, input_ids)

    # 2. 计算 KL 惩罚
    with torch.no_grad():
        ref_logits, _, _ = ref_model.model(input_ids, attention_mask)
        ref_log_probs = F.log_softmax(ref_logits, dim=-1)

    kl_div = compute_kl_penalty(log_probs, ref_log_probs)
    kl_penalty = kl_coef * kl_div.mean()

    # 3. 计算策略损失
    advantages_normalized = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
    policy_loss = compute_policy_loss(
        log_probs,
        old_log_probs,
        advantages_normalized,
        clip_ratio
    )

    # 4. 计算价值损失
    value_loss = compute_value_loss(values, returns)

    # 5. 总损失
    total_loss = policy_loss + vf_coef * value_loss + kl_penalty

    # 6. 反向传播
    optimizer.zero_grad()
    total_loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
    optimizer.step()

    return {
        'policy_loss': policy_loss.item(),
        'value_loss': value_loss.item(),
        'total_loss': total_loss.item(),
        'kl_div': kl_div.mean().item(),
    }
```

---

## DPO 完全手动实现

### 核心组件

#### 1. 计算 Log Probabilities

```python
def compute_log_probs(model, input_ids, attention_mask=None):
    """
    计算模型对输入序列的 log probabilities

    数学公式：
        log π(x) = log_softmax(logits)
        log_prob = Σ log π(x_t | x_<t)
    """
    # 前向传播获取 logits
    outputs = model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        return_dict=True
    )
    logits = outputs.logits

    # 计算 log_softmax
    log_probs = F.log_softmax(logits, dim=-1)

    # 收集每个位置实际 token 的 log 概率
    input_ids_expanded = input_ids.unsqueeze(-1)
    log_probs = log_probs.gather(2, input_ids_expanded).squeeze(-1)

    return log_probs
```

---

#### 2. DPO 损失函数

**数学公式**：

```
L_DPO = -log σ(β × ((log π_chosen - log π_ref_chosen)
                     - (log π_rejected - log π_ref_rejected)))
```

**手动实现**：

```python
def compute_dpo_loss(
    policy_log_probs_chosen: torch.Tensor,
    policy_log_probs_rejected: torch.Tensor,
    ref_log_probs_chosen: torch.Tensor,
    ref_log_probs_rejected: torch.Tensor,
    beta: float = 0.1
) -> torch.Tensor:
    """
    手动实现 DPO 损失函数

    直观理解：
        - (log π_chosen - log π_ref_chosen): 策略偏离参考的程度（chosen）
        - (log π_rejected - log π_ref_rejected): 策略偏离参考的程度（rejected）
        - 我们希望：chosen 的偏离 > rejected 的偏离
    """
    # 1. 计算 log 概率差（策略 vs 参考）
    chosen_logratios = policy_log_probs_chosen - ref_log_probs_chosen
    rejected_logratios = policy_log_probs_rejected - ref_log_probs_rejected

    # 2. 计算 DPO 的核心项
    logratios_diff = chosen_logratios - rejected_logratios

    # 3. 应用 sigmoid
    # sigmoid(x) = 1 / (1 + exp(-x))
    # 我们希望：chosen 的概率 > rejected 的概率
    # 即：logratios_diff > 0
    loss = -F.logsigmoid(beta * logratios_diff)

    return loss.mean()
```

---

#### 3. DPO 更新步骤

```python
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
    """
    # 1. 计算策略模型的 log_probs
    policy_log_probs_chosen = compute_log_probs(policy_model, **chosen_inputs)
    policy_log_probs_rejected = compute_log_probs(policy_model, **rejected_inputs)

    # 2. 计算参考模型的 log_probs（不需要梯度）
    with torch.no_grad():
        ref_log_probs_chosen = compute_log_probs(ref_model, **chosen_inputs)
        ref_log_probs_rejected = compute_log_probs(ref_model, **rejected_inputs)

    # 3. 平均序列长度
    policy_log_probs_chosen_mean = policy_log_probs_chosen.mean(dim=1)
    policy_log_probs_rejected_mean = policy_log_probs_rejected.mean(dim=1)
    ref_log_probs_chosen_mean = ref_log_probs_chosen.mean(dim=1)
    ref_log_probs_rejected_mean = ref_log_probs_rejected.mean(dim=1)

    # 4. 计算 DPO 损失
    loss = compute_dpo_loss(
        policy_log_probs_chosen_mean,
        policy_log_probs_rejected_mean,
        ref_log_probs_chosen_mean,
        ref_log_probs_rejected_mean,
        beta
    )

    # 5. 反向传播
    optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(policy_model.parameters(), max_norm=1.0)
    optimizer.step()

    # 6. 计算准确率
    with torch.no_grad():
        chosen_logratios = policy_log_probs_chosen_mean - ref_log_probs_chosen_mean
        rejected_logratios = policy_log_probs_rejected_mean - ref_log_probs_rejected_mean
        logratios_diff = chosen_logratios - rejected_logratios
        accuracy = (logratios_diff > 0).float().mean()

    return {
        'loss': loss.item(),
        'accuracy': accuracy.item(),
        'logratios_diff_mean': logratios_diff.mean().item(),
    }
```

---

## 核心概念详解

### 1. 优势函数 A_t

**定义**：

```python
A_t = Q(s_t, a_t) - V(s_t)
```

**物理意义**：

```
A_t = 这个动作比"平均水平"好多少

如果 A_t = +0.5：这个动作比平均好 0.5 分 → 应该增强
如果 A_t = -0.3：这个动作比平均差 0.3 分 → 应该减弱
如果 A_t =  0.0：这个动作很普通 → 保持不变
```

**GAE 计算**：

```python
A_t = δ_t + γλ×δ_{t+1} + (γλ)²×δ_{t+2} + ...

其中：
- δ_t = r_t + γ×V(s_{t+1}) - V(s_t)（TD 误差）
- γ = 0.99（折扣因子）
- λ = 0.95（GAE 参数）
```

---

### 2. 回报 R_t

**定义**：

```python
R_t = A_t + V(s_t)
```

**物理意义**：

```
R_t = 从状态 s_t 开始，实际能获得的总价值
     = 优势 + 基准价值
     = (动作带来的额外价值) + (状态本身的价值)
```

**训练价值网络**：

```python
# 价值网络的损失函数
loss = (R_t - V(s_t))²

# 意义：让 V(s_t) 接近 R_t
# 让预测的价值接近真实的回报
```

---

### 3. KL 惩罚

**定义**：

```python
KL_penalty = kl_coef × (ref_logprobs - logprobs)
          = β × KL(新模型 || 参考模型)
```

**物理意义**：

```
KL 散度衡量两个分布的差异

KL 惩罚的作用：
- 防止新策略偏离参考策略太远
- 保持训练稳定性
- 避免模式崩溃
```

---

### 4. 策略损失 vs 价值损失

| 维度         | 策略损失         | 价值损失      |
| ------------ | ---------------- | ------------- |
| **网络**     | 策略头           | 价值头        |
| **输入**     | 状态 s           | 状态 s        |
| **输出**     | 动作概率 π(a\|s) | 状态价值 V(s) |
| **标签来源** | 优势 A_t         | 回报 R_t      |
| **目标**     | 选好动作         | 预测价值      |
| **损失类型** | 策略梯度         | 回归 MSE      |

**为什么相加**：

```
总损失 = 策略损失 + 价值损失

类比围棋 AI：
- 策略网络：决定下哪一步
- 价值网络：评估当前局面
- 两个都要强，AI 才能赢
```

---

## 训练流程对比

### PPO vs DPO

| 维度           | PPO                                           | DPO                              |
| -------------- | --------------------------------------------- | -------------------------------- |
| **训练数据**   | 只有 prompt                                   | (prompt, chosen, rejected)       |
| **训练方式**   | 在线训练 (on-policy)                          | 离线训练 (off-policy)            |
| **模型架构**   | Policy + ValueHead + Reference + Reward Model | Policy + Reference               |
| **奖励信号**   | 需要训练好的 Reward Model                     | 不需要显式奖励模型               |
| **损失函数**   | L_CLIP + L_VF + KL_penalty                    | -log sigmoid(β × Δ)              |
| **训练稳定性** | 较复杂，需要调节多个超参数                    | 更稳定，超参数较少               |
| **计算成本**   | 较高（需要在线生成 + 训练多个模型）           | 较低（离线训练，只需要一个模型） |

---

### 数据需求对比

```python
# PPO: 只需要 prompt
PPO_data = ["什么是机器学习？"]

# DPO: 需要 (prompt, chosen, rejected)
DPO_data = {
    "prompt": "什么是机器学习？",
    "chosen": "机器学习是...",
    "rejected": "就是学习。"
}
```

---

### 损失函数对比

```python
# PPO: 复杂的多项损失
PPO_loss = policy_loss + value_loss + kl_penalty

# DPO: 简单的单项损失
DPO_loss = -log sigmoid(β × (chosen_logratio - rejected_logratio))
```

---

### 模型组件对比

```python
# PPO: 需要这些
PPO_components = [
    "Policy Model",
    "Value Head",
    "Reference Model",
    "Reward Model"
]

# DPO: 只需要这些
DPO_components = [
    "Policy Model",
    "Reference Model"  # 只用于计算 KL
]
```

---

## 使用指南

### 运行 PPO 手动实现

```bash
python ppo_manual.py
```

这会：

1. 加载 Qwen2.5-0.5B 模型
2. 添加价值头
3. 运行 PPO 训练
4. 打印每一步的统计信息

### 运行 DPO 手动实现

```bash
python dpo_manual.py
```

这会：

1. 打印 DPO vs PPO 对比表
2. 加载 Qwen2.5-0.5B 模型
3. 运行 DPO 训练
4. 打印详细的统计信息

---

## 关键参数说明

### PPO 参数

| 参数         | 默认值 | 说明                           |
| ------------ | ------ | ------------------------------ |
| `clip_ratio` | 0.2    | PPO 裁剪参数，控制策略更新幅度 |
| `vf_coef`    | 0.5    | 价值损失权重                   |
| `kl_coef`    | 0.1    | KL 惩罚系数                    |
| `gamma`      | 0.99   | 折扣因子                       |
| `lambda_gae` | 0.95   | GAE 参数                       |

### DPO 参数

| 参数   | 默认值 | 说明                                      |
| ------ | ------ | ----------------------------------------- |
| `beta` | 0.1    | KL 惩罚系数（控制模型偏离参考模型的程度） |

---

## 总结

本文档提供了 PPO 和 DPO 的完全透明实现，每个核心逻辑都有详细的数学公式和代码对应。主要特点：

1. **完全透明**：每个公式都对应代码，没有黑盒
2. **可调试性**：可以在任何地方插入 print 查看中间变量
3. **可修改性**：可以轻松修改任何部分（如 GAE 公式、损失函数等）
4. **教育性**：适合学习和理解强化学习的核心概念

通过这些实现，你可以：

- 深入理解 PPO 和 DPO 的工作原理
- 根据需要自定义和优化算法
- 调试和解决训练中的问题
- 将这些技术应用到实际项目中
