# RLVR 手动实现计划

## Context

需要实现一个RLVR（Reinforcement Learning with Verifiable Rewards）的手动实现，用于训练模型解决可验证奖励的任务（如数学题、代码等）。项目已有 `ppo_manual.py` 和 `dpo_manual.py`，风格一致。

**RLVR核心思想**：
- 不需要reward model，因为奖励可以直接验证（正确/错误）
- 类似于GRPO：对同一个prompt采样多个response，根据验证结果计算奖励
- 使用reference model做KL约束，防止策略偏离太远

## 实现文件

**新文件**: `/Users/Admin/Desktop/qc_fine_tuning/rlvr_manual.py`

## 核心组件

### 1. `PolicyWithValueHead` - 策略+价值头
- 复用 `ppo_manual.py` 的实现
- 简化：RLVR不需要value head，因为奖励是终态的

### 2. `ReferenceModel` - 参考模型（冻结）
- 复用 `ppo_manual.py` 的实现
- 用于计算KL散度约束

### 3. `verify_reward_function` - 可验证奖励函数
- 用户自定义的奖励验证逻辑
- 例如：数学题答案验证、代码执行结果验证
- 返回: reward (float, 通常0或1)

### 4. `compute_policy_loss` - 策略损失
```
L = -E[log π(a|s) * advantage]
```
简化为REINFORCE或PPO-style裁剪损失

### 5. `collect_rollouts` - 采样rollouts
- 对每个prompt，采样多个response
- 计算每个response的验证奖励

### 6. `update_step` - 更新步骤
- 计算策略损失 + KL损失
- 更新策略模型

## 训练流程

```
for epoch in range(num_epochs):
    for batch prompts:
        # 1. 采样多个responses
        responses = policy.generate(prompts, num_samples=4)

        # 2. 验证奖励
        rewards = [verify(p, r) for p, r in zip(prompts, responses)]

        # 3. 计算优势（用reward作为优势）
        advantages = normalize(rewards - baseline)

        # 4. 策略更新
        loss = compute_policy_loss(log_probs, advantages)
        loss += kl_coef * kl_penalty  # KL约束
        optimizer.step()

        # 5. Reference模型同步（可选）
        ref_model.load_state_dict(policy.state_dict())
```

## 关键设计决策

1. **ValueHead**: RLVR通常不需要，因为奖励是终态的（final reward）
2. **优势函数**: 直接用 `(reward - baseline)` 或相对排名
3. **Reference更新频率**: 每N步同步一次，或用EMA
4. **采样数量**: 每个prompt采样4-8个response

## 参考实现

- `ppo_manual.py` - PPO完整手动实现（含ValueHead、GAE、参考模型）
- `dpo_manual.py` - DPO完整手动实现（含log_probs计算、参考模型）

## 验证方式

1. `uv run python rlvr_manual.py` - 确认能运行无报错
2. 观察loss下降和reward提升趋势

---

## 实现状态

✅ **已完成** - `/Users/Admin/Desktop/qc_fine_tuning/rlvr_manual.py`

核心实现：
- `PolicyModel`: 纯策略网络（无ValueHead）
- `ReferenceModel`: 冻结参考模型
- `compute_advantages_group_relative`: GRPO风格排名优势
- `compute_policy_loss`: REINFORCE风格策略损失
- `collect_rollouts`: 采样多个response并验证奖励
- `rlvr_update_step`: 完整更新循环

验证：语法检查通过 ✅
