# PPO 损失函数详解

## 一、PPO 综述

PPO（Proximal Policy Optimization，近端策略优化）是一种强化学习算法，用于训练语言模型生成更好的回答。

### 核心思想

PPO 通过"生成 → 打分 → 更新"的循环，让模型学会产生高奖励的回答。

### 三类模型

| 模型 | 作用 | 是否更新 |
|------|------|---------|
| **Policy + ValueHead** | 生成文本 + 预测价值 | ✓ 更新 |
| **Reference** | 冻结的原始模型 | ✗ 冻结 |
| **Reward Model** | 给回答打分 | ✗ 冻结 |

### 完整训练流程


```
┌─────────────────────────────────────────────────────────┐
│ 第1步：收集经验（Rollout）                               │
│                                                         │
│  for 每个 query:                                        │
│    1. 用当前策略生成 response                           │
│    2. Reward Model 给 response 打分                     │
│    3. 记录: log_probs, values, rewards                 │
│    4. 用 GAE 计算: advantages, returns                │
└─────────────────────────────────────────────────────────┘
                          ↓
┌─────────────────────────────────────────────────────────┐
│ 第2步：PPO 更新（多轮）                                 │
│                                                         │
│  for epoch in range(ppo_epochs=4):                    │
│    1. 前向传播 → 新 log_probs, 新 values               │
│    2. 计算 KL 惩罚（防止跑太远）                        │
│    3. 计算策略损失 + 价值损失 + KL                      │
│    4. 反向传播更新模型                                  │
└─────────────────────────────────────────────────────────┘
                          ↓
┌─────────────────────────────────────────────────────────┐
│ 第3步：重复                                             │
│  用更新后的模型继续收集经验 → 更新 → 收集经验...        │
└─────────────────────────────────────────────────────────┘
```

---

## 二、三种损失函数概述

用"学开车"来理解三种损失：

| 损失 | 比喻 | 问的问题 |
|------|------|---------|
| **策略损失** | 教练的指导 | "你现在踩油门对不对？" |
| **价值损失** | 预估到达时间 | "从这里到目的地，还要开多久？" |
| **KL损失** | 驾校教的 basics | "别把刹车油门搞混了" |

---

## 三、前置条件：rewards 和 values 从哪来

在 GAE 计算 advantage 之前，需要先得到 `rewards` 和 `values`。这两个数据来自 PPO 的**经验收集阶段**（`collect_rollouts`）。

### values 从哪来

values 来自**策略模型**的 ValueHead：

```
策略模型.forward(input_ids)
        ↓
Transformer 主干输出 hidden_states
        ↓
ValueHead(hidden_states) → values [batch, seq_len]
```

> 详见 `ppo_manual.py` 第 29-93 行 `PolicyWithValueHead`

```python
# 模型结构
class PolicyWithValueHead(nn.Module):
    def __init__(self, base_model_name):
        self.base_model = AutoModelForCausalLM.from_pretrained(...)  # 预训练模型
        hidden_size = self.base_model.config.hidden_size
        self.value_head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.Tanh(),
            nn.Linear(hidden_size // 2, 1)  # 输出标量 value
        )

    def forward(self, input_ids, attention_mask=None):
        outputs = self.base_model.model(input_ids=input_ids, ...)
        hidden_states = outputs.last_hidden_state  # [batch, seq_len, hidden_size]
        logits = self.base_model.lm_head(hidden_states)  # 策略输出
        values = self.value_head(hidden_states).squeeze(-1)  # 价值输出
        return logits, hidden_states, values
```

#### 矩阵结构变化详解

Transformer 主干输出的 `last_hidden_state` 是多层神经网络最后一层的隐藏状态，形状为 `[batch, seq_len, hidden_size]`：

| 变量 | 形状 | 说明 |
|------|------|------|
| `hidden_states` | `[batch, seq_len, hidden_size]` | Transformer 输出的隐藏状态 |
| `lm_head` | `Linear(hidden_size, vocab_size)` | 语言模型头，将隐藏状态映射到词表大小 |
| `logits` | `[batch, seq_len, vocab_size]` | 每个位置对词表中每个词的打分（未归一化） |
| `value_head` | `Sequential(Linear(hidden_size, hidden_size//2), Tanh, Linear(hidden_size//2, 1))` | 价值头 |
| `values` (处理前) | `[batch, seq_len, 1]` | squeeze 前 |
| `values` (处理后) | `[batch, seq_len]` | `squeeze(-1)` 后，去掉多余的维度 |

**直观理解**：

`last_hidden_state` 就像 Transformer 这个多层神经网络最后一层输出的"记忆向量"：
- 形状 `[batch, seq_len, hidden_size]` 可以理解为：每句话的每个字都用一个多位数向量来表示
- `batch` = 一次处理几句话
- `seq_len` = 每句话有多少个字
- `hidden_size` = 每个字用多少个数字描述

**LM Head** 就是一个矩阵乘法（线性变换），把每个字的多位数向量映射成一个词表大小的向量——这个向量里每个数代表"这个词的可能性"，这就是 logits。

**Value Head** 也是一个矩阵乘法，但把向量映射成一个数——代表"这个位置值多少分"。

`squeeze(-1)` 就是把 `[batch, seq_len, 1]` 变成 `[batch, seq_len]`，去掉多余的维度。

### rewards 从哪来

rewards 来自 **Reward Model** 对回答的打分：

```
query + response → Reward Model → reward 分数
```

> 详见 `ppo_manual.py` 第 401-430 行 `SimpleRewardModel`

```python
class SimpleRewardModel:
    """简化版 Reward Model"""
    def __call__(self, query: str, response: str) -> float:
        reward = 0.5  # 基础分
        reward += min(len(response) / 200, 0.3)  # 长度奖励
        # 关键词奖励
        for keyword in ['learn', 'explain', 'example']:
            if keyword in response.lower():
                reward += 0.04
        return min(reward, 1.0)
```

### 完整经验收集流程

> 详见 `ppo_manual.py` 第 468-583 行 `collect_rollouts`

```
1. query → tokenizer → token_ids
        ↓
2. model.generate(token_ids) → response_ids（策略模型生成回答）
        ↓
3. 拼接 query + response → full_input_ids
        ↓
4. model.forward(full_input_ids) → values（ValueHead 输出）
        ↓
5. reward_model(query, response_text) → reward（Reward Model 打分）
        ↓
6. rewards, values → compute_advantages_gae → advantages, returns
```

---

## 四、GAE 优势估计

### GAE 是什么

GAE（Generalized Advantage Estimation，广义优势估计）是一种计算 **advantage（优势）** 的方法。

**GAE 不是损失函数**，而是策略损失的**输入**。

### 关系梳理

```
rewards, values
       ↓
    GAE（计算 advantage）
       ↓
   advantage
       ↓
策略损失（需要 advantage 作为输入）
```

### 核心公式

```
δ_t = r_t + γ × V(s_{t+1}) - V(s_t)    # TD 误差
A_t = δ_t + γλ × δ_{t+1} + (γλ)² × δ_{t+2} + ...  # GAE
```

| 符号 | 含义 |
|------|------|
| δ_t | TD 误差：当前奖励 + 下一个状态价值 - 当前状态价值 |
| γ (gamma) | 折扣因子，考虑远期奖励的重要性 |
| λ (lambda) | 控制偏差-方差权衡 |
| A_t | 优势函数：当前动作比平均好多少 |

### 逐步拆解

#### 第一步：计算 TD 误差 

```python
delta = r_t + gamma * values[:, t+1] - values[:, t]
```

- 如果 **r_t 高** 且 **V(s_{t+1}) 高** → δ 为正，动作好
- 如果 **r_t 低** 且 **V(s_{t+1}) 低** → δ 为负，动作差

#### 第二步：累积 TD 误差

```python
advantage = delta + gamma * lambda_gae * advantage  # 从后往前累积
```

从序列末端向前递归累积，越前面的 λ 越多方（考虑更多后续步骤）。

### γ 和 λ 参数的作用

| 参数 | 值 | 效果 |
|------|-----|------|
| γ (gamma) | 0.99 | 远期奖励打 99 折，模型更重视眼前 |
| λ (lambda) | 0.95 | 平衡偏差和方差 |

| λ 值 | 效果 |
|------|------|
| **λ 接近 0** | 更相信价值网络，方差小，但可能有偏差 |
| **λ 接近 1** | 更相信实际 reward，偏差小，但方差大 |

PPO 通常用 **λ=0.95** 作为平衡。

### advantage 和 returns

```python
returns = advantages + values  # R_t = A_t + V(s_t)
```

| 变量 | 含义 | 用途 |
|------|------|------|
| advantages | 动作比平均好多少 | 策略损失的核心输入 |
| returns | 折扣累积奖励 + 基线 | 价值损失的目标 |

### 代码实现

> 详见 `ppo_manual.py` 第 158-204 行 `compute_advantages_gae`

```python
def compute_advantages_gae(rewards, values, gamma=0.99, lambda_gae=0.95):
    batch_size, seq_len = rewards.shape
    advantages = torch.zeros_like(rewards)
    returns = torch.zeros_like(rewards)

    for i in range(batch_size):
        advantage = 0
        for t in reversed(range(seq_len)):
            if t == seq_len - 1:
                delta = rewards[i, t] - values[i, t]
            else:
                delta = rewards[i, t] + gamma * values[i, t + 1] - values[i, t]
            advantage = delta + gamma * lambda_gae * advantage
            advantages[i, t] = advantage
        returns[i] = advantages[i] + values[i]

    return advantages, returns
```

---

## 五、策略损失 (Policy Loss) — 核心

策略损失回答的问题是：**"这个动作值不值得选？"**

> 详见 `ppo_manual.py` 第 211-250 行 `compute_policy_loss`

### 核心公式

```python
ratio = exp(log π_new - log π_old)
policy_loss = -min(A × ratio, A × clip(ratio, 1-ε, 1+ε))
```

### 逐步拆解

#### 第一步：计算 ratio（概率比）

```python
ratio = exp(log π_new - log π_old)
```

| 符号 | 含义 |
|------|------|
| π_new | 新策略选中这个动作的概率 |
| π_old | 旧策略选中这个动作的概率 |
| ratio | 新/旧概率的比值 |

**举例**：
- 旧策略：选这个动作的概率 = 1%
- 新策略：选这个动作的概率 = 5%
- ratio = 5% / 1% = **5**

意味着新策略选中这个动作的可能性是旧策略的5倍。

#### 第二步：裁剪 ratio

```python
clip(ratio, 1-ε, 1+ε)  # ε = 0.2
```

把 ratio 限制在 [0.8, 1.2] 范围内：

| ratio 原始值 | clip 后 |
|-------------|--------|
| 0.5 | 0.8（被拉高） |
| 1.0 | 1.0（不变） |
| 2.0 | 1.2（被压低） |
| 5.0 | 1.2（被压低） |

#### 第三步：取 min

```python
policy_loss = -min(A × ratio, A × clip(ratio, 1-ε, 1+ε))
```

**举例一**：动作很好，A = +10（advantage 为正）

| 情况 | ratio | A×ratio | A×clip | min | -min |
|------|-------|---------|--------|-----|------|
| 大幅增加概率 | 5 | 50 | 12 | 12 | -12 |
| 小幅增加概率 | 1.1 | 11 | 11 | 11 | -11 |

**分析**：大幅增加概率时，损失从 50 被限制到 12，限制了更新幅度。

**举例二**：动作很差，A = -10（advantage 为负）

| 情况 | ratio | A×ratio | A×clip | min | -min |
|------|-------|---------|--------|-----|------|
| 大幅减少概率 | 0.2 | -2 | -12 | -12 | +12 |
| 小幅减少概率 | 0.9 | -9 | -9 | -9 | +9 |

**分析**：如果策略反而增加了差动作的概率，损失会很大（鼓励避免这种情况）。

### 为什么要裁剪？

**核心目的**：防止一次更新幅度过大，导致策略崩溃。

```
没裁剪：概率比 5 → 损失很小 → 鼓励继续增大 → 可能失控
裁剪后：概率比 5 → 限制到 1.2 → 损失变大 → 温和鼓励
```

**类比**：
- **没裁剪**：你考试进步了 50 分，老师说"太棒了继续保持！"
  → 你飘了，下次退步 30 分
- **裁剪后**：老师会说"进步了，但最多算你进步 20 分，稳住！"
  → 稳步提升，不飘不躁

### 一句话记忆

```
L_policy = -min( 好处上限, 好处实际 )
```

**好处上限**：最多让你增长 1+ε = 1.2 倍
**好处实际**：按真实概率比计算

---

## 六、价值损失 (Value Loss)

价值损失回答的问题是：**"这个局面最终值多少分？"**

> 详见 `ppo_manual.py` 第 253-272 行 `compute_value_loss`

### 核心公式

```python
value_loss = 0.5 × (V(s) - R)²
```

| 符号 | 含义 |
|------|------|
| V(s) | 模型预测的分数（当前状态价值） |
| R | 实际回报（从当前状态最终得到的分数） |

### 举例

- 模型预测这局面值 **8 分**
- 实际回报是 **10 分**
- 损失 = 0.5 × (8-10)² = **2**

### 作用

让估值越来越准，配合策略做更好的决策。

如果估值偏低，模型会低估好动作的价值；如果估值偏高，模型会高估差动作的价值。通过不断修正 V(s) 接近 R，模型学会准确评估每个状态的价值。

---

## 七、KL 损失 (KL Penalty)

KL 损失回答的问题是：**"别把预训练学到的能力忘光了？"**

> 详见 `ppo_manual.py` 第 275-296 行 `compute_kl_penalty`

### 核心公式

```python
KL = Σ π(a|s) × log(π(a|s) / π_ref(a|s))
```

衡量新策略与参考模型的差异。

### 作用

防止策略完全偏离预训练能力，保持模型基本能力不退化。

---

## 八、三种损失如何配合

> 完整的 PPO 更新步骤见 `ppo_manual.py` 第 303-394 行 `ppo_update_step`

### 用"学开车"完整比喻

```
1. 你看到路况（当前状态 s）
   ↓
2. 你决定踩油门（选择动作 a）
   ↓
3. 价值网络说："这个操作，预计能开100米才停车"（V(s)）
   ↓
4. 你踩了油门，实际开了80米（实际回报 R）
   ↓
5. 比较：80 vs 100 → 价值损失 = 0.5 × (100-80)² = 2
   ↓
6. 同时教练喊："你踩油门的时机比上次好！"（advantage = +20）
   ↓
7. 策略损失 = -min(A × ratio, A × clip(ratio))
   ↓
8. 副驾驶提醒："别忘了驾校教的，别压线"（KL penalty）
```

### 总损失公式

```python
total_loss = 策略损失 + 0.5 × 价值损失 + 0.1 × KL损失
```

| 损失 | 权重 | 原因 |
|------|------|------|
| 策略损失 | 1.0 | **最重要**，直接决定决策质量 |
| 价值损失 | 0.5 | **次要**，辅助决策 |
| KL损失 | 0.1 | **约束**，别跑太远 |

### 协作图

```
                    ┌─────────────────────────────────────┐
                    │           状态 s                    │
                    └─────────────────────────────────────┘
                                       │
           ┌───────────────────────────┼───────────────────────────┐
           ↓                           ↓                           ↓
    ┌──────────────┐          ┌──────────────┐          ┌──────────────┐
    │   价值网络    │          │   策略网络    │          │   参考网络    │
    │   V(s) 预测   │          │   选动作      │          │   基础能力    │
    └──────────────┘          └──────────────┘          └──────────────┘
           │                           │                           │
           ↓                           ↓                           ↓
    价值损失 MSE                策略损失 PPO                KL散度
    (估准了吗？)               (选对了吗？)                (别忘本)
           │                           │                           │
           └───────────────────────────┼───────────────────────────┘
                                       ↓
                              总损失 = 三者相加
                                       ↓
                              反向传播更新模型
```

---

## 九、一句话总结

- **策略损失**：指挥你"往哪走"——让好动作的概率上升
- **价值损失**：告诉你"走了多远"——让估值越来越准
- **KL损失**：提醒你"别忘了基本功"——保持预训练能力

三者一起配合，让模型学会做出更好的决策，同时不忘老本行。

---

## 附录：代码实现索引

| 功能 | 文件位置 | 函数名 |
|------|---------|--------|
| GAE 优势估计 | `ppo_manual.py` 第 158-204 行 | `compute_advantages_gae` |
| 策略损失 | `ppo_manual.py` 第 211-250 行 | `compute_policy_loss` |
| 价值损失 | `ppo_manual.py` 第 253-272 行 | `compute_value_loss` |
| KL 损失 | `ppo_manual.py` 第 275-296 行 | `compute_kl_penalty` |
| PPO 更新步骤 | `ppo_manual.py` 第 303-394 行 | `ppo_update_step` |
| 策略+价值头 | `ppo_manual.py` 第 29-98 行 | `PolicyWithValueHead` |

