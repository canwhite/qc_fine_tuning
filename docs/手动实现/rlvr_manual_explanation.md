# RLVR 训练流程详解

## 一、整体架构

RLVR（Reinforcement Learning with Verifiable Rewards）的核心思想是：**奖励不需要估计，可以在终态直接验证**。比如数学题，对错一目了然，不需要像 PPO 那样训练一个 value network 去估计"中间状态值"。

整个训练循环分为三个阶段：**采样 → 优势计算 → 更新**。

## 二、采样阶段（collect_rollouts）

对每个 prompt，模型生成 N 个不同的 response（通过 `do_sample=True` + temperature）。
每个 response 都由 `verify_reward(prompt, response)` 给出 0 或 1 的奖励。

收集的数据结构：
- `input_ids`：完整序列（prompt + response）
- `response_mask`：标记 response 部分（用于后续只计算 response 的损失）
- `rewards`：每个 response 的验证奖励
- `advantages`：基于排名的相对优势

## 三、优势计算（compute_advantages_group_relative）

### 3.1 为什么需要优势而不是直接用奖励？

奖励（reward）描述的是"这个 response 最终好不好"，但我们希望知道的是"这个 action 相对于平均水平好多少"——这就是优势（advantage）。

例如：同一 prompt 下 4 个 response 的奖励可能是 `[0, 0, 1, 0]`。光看奖励只知道第三个好，但不知道其他三个有多差。优势则表达了"比其他三个好多少"。

### 3.2 GRPO 风格的排名优势

优势计算在组内进行。每个 prompt 及其采样的 N 个 response 构成一个组，按奖励排名分配优势：

```
ranks = [0, 1, 2, 3]        # 0=最低奖励, 3=最高奖励
normalized = ranks / (N-1)  # → [0, 0.33, 0.67, 1.0]
advantages = normalized - 0.5  # → [-0.5, -0.17, +0.17, +0.5]
```

**核心代码**（rlvr_manual.py 第 147-163 行）：

```python
def compute_advantages_group_relative(rewards, num_samples):
    batch_size = rewards.shape[0] // num_samples
    advantages = torch.zeros_like(rewards)

    for i in range(batch_size):
        start = i * num_samples
        end = start + num_samples
        group_rewards = rewards[start:end]

        # 计算排名：排序后的原始索引位置
        sorted_indices = torch.argsort(group_rewards)
        ranks = torch.argsort(sorted_indices).float()  # 0=最低, num_samples-1=最高

        # 归一化到 [-0.5, 0.5]
        normalized_ranks = (ranks / (num_samples - 1)) - 0.5
        advantages[start:end] = normalized_ranks

    return advantages
```

### 3.3 优势的数据维度

假设 `batch_size=2, num_samples=4`，则 `rewards` 和 `advantages` 的 shape 都是 `[8]`：

```
prompt[0] 的 4 个 response: advantages[0:4]
prompt[1] 的 4 个 response: advantages[4:8]
```

同一 prompt 组内的优势之和为零：`+0.5 + 0.1667 + (-0.1667) + (-0.5) = 0`

### 3.4 边界情况

当同一组内所有奖励相同时（均为 0 或均为 1），按排名仍有优势分配。这在语义上不够干净——全部答错时仍会强制区分排名靠前和靠后的 response。更严谨的做法是对奖励全相同的组直接赋予零优势：

```python
if grp.max() == grp.min():
    advantages[start:end] = 0.0
    continue
```

## 四、损失计算（核心）

总损失 = **策略损失** + β × **KL 惩罚**

```
total_loss = policy_loss + kl_coef * kl_penalty
```

两者承担不同职责：**策略损失推动策略学习有用行为，KL 惩罚阻止策略走太远**。

### 4.1 策略损失（compute_policy_loss）

REINFORCE 风格，目标是最大化优势加权的 log 概率：

```
L_policy = -E[log π(a|s) × advantage]
```

**公式中的符号含义**：
- **E** = 期望值，即"对所有情况求平均"
- **a|s** = 在状态 s 下选择动作 a，读作"给定 s 的 a"
- 文本生成语境下：s 是当前上下文（prompt + 已生成 token），a 是下一个要生成的 token
- **π(a|s)** = 策略网络在上下文 s 下输出 token a 的概率
- **advantage** = 这个 action 相对同组的好坏程度

```
L_policy = -mean(log π(a_i|s_i) × advantage_i)
```

实际代码中 `log π(a|s)` 就是 `mean_log_probs[i]`，`advantage` 就是 `advantages[i]`，两者按索引相乘后取负均值。

**关键连接：advantages 和 mean_log_probs 如何对应**

假设 `batch_size=2, num_samples=4`，共 8 个 sample。优势计算后：

```
# advantages（每组内: 最高=+0.5, 次高=+0.167, 次低=-0.167, 最低=-0.5）
advantages = [
    +0.500,  # prompt[0] response[排名1] → 高奖励
    +0.167,  # prompt[0] response[排名2]
    -0.167,  # prompt[0] response[排名3]
    -0.500,  # prompt[0] response[排名4] → 低奖励
    +0.500,  # prompt[1] response[排名1]
    +0.167,  # prompt[1] response[排名2]
    -0.167,  # prompt[1] response[排名3]
    -0.500,  # prompt[1] response[排名4]
]
```

每个 sample i 的 `mean_log_probs[i]` 是该 response 内所有 token 的 log 概率平均值（标量）。两者通过**索引直接对应**：

| sample i | mean_log_probs[i] | advantages[i] | 乘积 | 损失贡献 |
|---------|-------------------|--------------|------|---------|
| 0 | -1.2 | +0.5 | -0.60 | +0.60（鼓励增加） |
| 3 | -2.3 | -0.5 | +1.15 | -1.15（惩罚增加） |
| ... | ... | ... | ... | ... |

**损失函数就是所有样本乘积的负均值**：

```
L_policy = -mean(advantages[i] × mean_log_probs[i]),  i = 0...7
```

**步骤分解**：

1. **前向传播**：模型输出 logits，计算 `log_probs`（每个 token 位置对实际 token 的 log 概率）

   ```python
   logits = model(input_ids, attention_mask)  # [8, seq_len, vocab_size]
   log_probs = F.log_softmax(logits, dim=-1)
   ```

2. **屏蔽 prompt 部分**：用 `response_mask` 只保留 response 位置的 log_prob

   ```python
   response_log_probs = log_probs * response_mask  # [8, seq_len]
   ```

3. **取平均**：对每个 sample，将 response 内所有 token 的 log_prob 求平均

   ```python
   response_mask_sum = response_mask.sum(dim=1) + 1e-8
   mean_log_probs = response_log_probs.sum(dim=1) / response_mask_sum
   # shape: [8] — 每个 sample 一个标量
   ```

4. **加权合并**：这是 advantages 和 log_probs 真正汇合的地方

   ```python
   # advantages: [8], mean_log_probs: [8]，按索引逐元素相乘
   weighted = mean_log_probs * advantages  # [8]
   policy_loss = -weighted.mean()          # 标量
   ```

**直观理解**：

| advantages 值 | 梯度方向 | 效果 |
|-------------|---------|------|
| 正（+0.5） | 推高 mean_log_probs | 增加这类 response 出现的概率 |
| 负（-0.5） | 压低 mean_log_probs | 减少这类 response 出现的概率 |

### 4.2 KL 惩罚（compute_kl_penalty）

防止策略在一次更新中偏离参考模型太远。使用简化的 KL 散度近似：

```
L_kl = Σ (log π_new[a] - log π_ref[a])   # 只在 response 位置
```

**为什么用简化的 KL**：

完整 KL 散度是 `Σ π_ref[a] × (log π_ref[a] - log π_new[a])`，计算需要整个词表分布。简化版本只关注实际选中 token 的 log prob 差异，计算更高效，在实践中效果相近。

**关键实现细节**：需要把词表维度的 log prob `gather` 到 per-token 位置：

```python
def compute_kl_penalty(logits, ref_log_probs, input_ids, response_mask):
    new_log_probs = F.log_softmax(logits, dim=-1)

    # gather：每个位置取实际 token 的 log prob
    # [batch, seq_len, vocab] → [batch, seq_len]
    new_gathered = new_log_probs.gather(2, input_ids.unsqueeze(2)).squeeze(2)
    ref_gathered = ref_log_probs.gather(2, input_ids.unsqueeze(2)).squeeze(2)

    token_kl = new_gathered - ref_gathered  # [batch, seq_len]
    masked_kl = token_kl * response_mask

    return masked_kl.sum(dim=1).mean()
```

**KL > 0** 意味着新策略在 response 位置的平均 log prob 高于参考策略（偏离）；**KL ≈ 0** 表示几乎没有偏离。

### 4.3 两者如何联合工作

策略损失和 KL 损失在同一个优化目标中相加，梯度同时来自两个信号：

```
total_loss = policy_loss + kl_coef * kl_penalty
```

**两者梯度方向可能冲突**：

- 策略损失想把策略推向奖励更高的 response（可能显著偏离初始策略）
- KL 惩罚想把策略拉回参考模型（维持稳定）

**β（kl_coef）的作用**：

```
β 太小 → KL 惩罚无效，策略可能一步更新过大，走向不稳定
β 太大 → 策略被锁死在参考模型附近，学不到新东西
```

典型的 β 在 0.01 ~ 0.3 之间，需要根据具体任务调参。

**梯度叠加示意**：

```
                    total_loss
                        │
            ┌───────────┴───────────┐
            ↓                       ↓
    policy_loss              kl_coef * kl_penalty
            │                       │
            ↓                       ↓
    鼓励高奖励 response      拉回参考模型
            │                       │
            └───────────┬───────────┘
                        ↓
              实际梯度 = 两者加和
```

在 rlvr_manual.py 的实现中，两者合并后通过 `total_loss.backward()` 一次性更新：

```python
total_loss = policy_loss + kl_coef * kl_penalty
optimizer.zero_grad()   # 清除上一轮累加的梯度
total_loss.backward()  # 反向传播计算梯度
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)  # 裁剪防止梯度爆炸
optimizer.step()       # 用裁剪后的梯度更新参数
```

## 五、更新步骤（rlvr_update_step）

```python
def rlvr_update_step(model, ref_model, optimizer, batch, kl_coef):
    input_ids = batch['input_ids']
    attention_mask = batch['attention_mask']
    response_mask = batch['response_mask']
    advantages = batch['advantages']

    # 1. 前向传播获取 log_probs
    logits = model(input_ids, attention_mask)
    log_probs = compute_log_probs(logits, input_ids)

    # 2. 参考模型的 log_probs（冻结）
    ref_log_probs = ref_model.get_log_probs(input_ids, attention_mask)

    # 3. 计算两部分损失
    policy_loss = compute_policy_loss(log_probs, advantages, response_mask)
    kl_penalty = compute_kl_penalty(logits, ref_log_probs, input_ids, response_mask)

    # 4. 合并反向传播
    total_loss = policy_loss + kl_coef * kl_penalty
    optimizer.zero_grad()   # 清除上一轮累加的梯度（PyTorch 反向传播是累加的，不清零会导致梯度方向错误）
    total_loss.backward()  # 反向传播计算梯度
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)  # 裁剪防止梯度爆炸
    optimizer.step()       # 用裁剪后的梯度更新参数

    # 5. 同步参考模型（每次更新后立即同步）
    ref_model.model.load_state_dict(model.state_dict())

    return {
        'policy_loss': policy_loss.item(),
        'kl_penalty': kl_penalty.item(),
        'total_loss': total_loss.item(),
    }
```

## 六、完整训练流程图

```
prompt → 模型采样N个response → verify_reward计算奖励 → rewards
              ↓
        compute_advantages_group_relative → advantages
              ↓
        当前策略前向传播 → log_probs
        参考模型前向传播 → ref_log_probs
              ↓
        advantages × log_probs → policy_loss
        (log_probs - ref_log_probs) → kl_penalty
              ↓
        total_loss = policy_loss + β × kl_penalty
              ↓
        backward + clip + step
              ↓
        同步参考模型
              ↓
        重复直到收敛
```

## 七、关键设计总结

| 设计 | 原因 |
|------|------|
| 不需要 value network | 奖励终态可验证，直接用排名优势 |
| response_mask | 只对 response 部分计算梯度，prompt 不受影响 |
| 每次更新后同步参考模型 | 确保 KL 惩罚始终反映真实的偏离程度 |
| 梯度裁剪 max_norm=1.0 | 防止单步更新过大导致崩溃 |
| GRPO 排名优势 | 不依赖绝对奖励值，只比较组内相对好坏 |
