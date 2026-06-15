# DPO 损失函数详解

## 一、DPO 综述

DPO（Direct Preference Optimization，直接偏好优化）是一种比 PPO 更简单的偏好对齐方法。

### 核心思想

DPO 不需要奖励模型、不需要价值网络、不需要在线生成，直接用人类标注的偏好数据训练。

### 数据格式

DPO 的训练数据是 **(prompt, chosen, rejected)** 三元组：

| 字段 | 含义 |
|------|------|
| prompt | 问题 |
| chosen | 人类选择的好回答 |
| rejected | 人类拒绝的差回答 |

### DPO vs PPO

| 对比项 | PPO | DPO |
|--------|-----|-----|
| **数据** | 只有 prompt，模型在线生成回答 | prompt + chosen + rejected（离线） |
| **训练方式** | 在线 (on-policy) | 离线 (off-policy) |
| **模型架构** | Policy + ValueHead + Reference + Reward | Policy + Reference |
| **奖励模型** | 需要 | 不需要 |
| **复杂度** | 高 | 低 |
| **稳定性** | 较差 | 更稳定 |

---

## 二、DPO 训练流程

```
┌─────────────────────────────────────────────────────────┐
│ 输入：(prompt, chosen, rejected) 三元组                  │
└─────────────────────────────────────────────────────────┘
                          ↓
┌─────────────────────────────────────────────────────────┐
│ 1. 编码：将 prompt+chosen 和 prompt+rejected 分别拼接    │
│    → "prompt + chosen" → token_ids_chosen              │
│    → "prompt + rejected" → token_ids_rejected           │
└─────────────────────────────────────────────────────────┘
                          ↓
┌─────────────────────────────────────────────────────────┐
│ 2. 前向传播                                            │
│    → 策略模型：计算 chosen 和 rejected 的 log_probs    │
│    → 参考模型：计算 chosen 和 rejected 的 log_probs     │
│    （参考模型冻结，不更新）                             │
└─────────────────────────────────────────────────────────┘
                          ↓
┌─────────────────────────────────────────────────────────┐
│ 3. 计算 DPO 损失                                       │
│    → 比较策略模型对 chosen vs rejected 的偏好程度       │
│    → 考虑与参考模型的偏离                               │
└─────────────────────────────────────────────────────────┘
                          ↓
┌─────────────────────────────────────────────────────────┐
│ 4. 反向传播更新策略模型                                 │
└─────────────────────────────────────────────────────────┘
```

---

## 三、前置条件：log_probs 从哪来

在 DPO 计算损失之前，需要先得到 `policy_log_probs` 和 `ref_log_probs`。

### 策略模型 vs 参考模型

| 模型 | 来源 | 是否更新 |
|------|------|---------|
| **策略模型** | `AutoModelForCausalLM.from_pretrained(model_name)` | ✓ 更新 |
| **参考模型** | `AutoModelForCausalLM.from_pretrained(model_name)`，然后冻结 | ✗ 冻结 |

> 详见 `dpo_manual.py` 第 429-451 行

```python
# 策略模型（会更新）
policy_model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.float16,
    trust_remote_code=True
).to(device)

# 参考模型（冻结）
ref_model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.float16,
    trust_remote_code=True
).to(device)
ref_model.eval()
for param in ref_model.parameters():
    param.requires_grad = False  # 冻结
```

### log_probs 的计算

> 详见 `dpo_manual.py` 第 31-68 行 `compute_log_probs`

```python
def compute_log_probs(model, input_ids, attention_mask=None):
    # 1. 前向传播获取 logits
    outputs = model(input_ids=input_ids, attention_mask=attention_mask)
    logits = outputs.logits  # [batch, seq_len, vocab_size]

    # 2. 计算 log_softmax
    log_probs = F.log_softmax(logits, dim=-1)  # [batch, seq_len, vocab_size]

    # 3. 收集每个位置实际 token 的 log 概率
    log_probs = log_probs.gather(2, input_ids.unsqueeze(-1)).squeeze(-1)  # [batch, seq_len]
    return log_probs
```

**直观理解**：
- `logits` 是模型对每个位置选择每个词的打分（未归一化）
- `F.log_softmax` 把打分转成 log 概率
- `.gather` 把每个位置对应的 token 的 log 概率挑出来

### DPO 更新流程中的 log_probs 计算

> 详见 `dpo_manual.py` 第 217-238 行 `dpo_update_step`

```python
# 策略模型的 log_probs（需要梯度）
policy_log_probs_chosen = compute_log_probs(policy_model, **chosen_inputs)
policy_log_probs_rejected = compute_log_probs(policy_model, **rejected_inputs)

# 参考模型的 log_probs（不需要梯度，用 torch.no_grad()）
with torch.no_grad():
    ref_log_probs_chosen = compute_log_probs(ref_model, **chosen_inputs)
    ref_log_probs_rejected = compute_log_probs(ref_model, **rejected_inputs)

# 对序列长度求平均（简化处理）
policy_log_probs_chosen_mean = policy_log_probs_chosen.mean(dim=1)
policy_log_probs_rejected_mean = policy_log_probs_rejected.mean(dim=1)
ref_log_probs_chosen_mean = ref_log_probs_chosen.mean(dim=1)
ref_log_probs_rejected_mean = ref_log_probs_rejected.mean(dim=1)
```

---

## 四、DPO 损失函数详解

### 核心公式

```
L_DPO = -log σ(β × ((log π_chosen - log π_ref_chosen)
                    - (log π_rejected - log π_ref_rejected)))
```

其中：
- π_chosen：策略模型对 chosen 的概率
- π_rejected：策略模型对 rejected 的概率
- π_ref_chosen：参考模型对 chosen 的概率
- π_ref_rejected：参考模型对 rejected 的概率
- β：KL 惩罚系数
- σ：sigmoid 函数

---

## 五、逐步拆解 DPO 损失

### 第一步：计算 log 概率差

```python
# 策略相比参考，对 chosen 的偏离程度
chosen_logratios = policy_log_probs_chosen - ref_log_probs_chosen

# 策略相比参考，对 rejected 的偏离程度
rejected_logratios = policy_log_probs_rejected - ref_log_probs_rejected
```

| 符号 | 含义 |
|------|------|
| chosen_logratios > 0 | 策略比参考更倾向 chosen |
| chosen_logratios < 0 | 策略比参考更远离 chosen |

### 第二步：计算差值

```python
logratios_diff = chosen_logratios - rejected_logratios
```

这是 DPO 的核心：我们希望 chosen 的偏离程度 > rejected 的偏离程度。

### 第三步：应用 sigmoid

```python
sigmoid_logratios = torch.sigmoid(beta * logratios_diff)
```

sigmoid 函数将任意值映射到 (0, 1)：

| logratios_diff | sigmoid(β×logratios_diff) |
|---------------|-------------------------|
| 大于 0 | 接近 1（损失小） |
| 小于 0 | 接近 0（损失大） |
| 等于 0 | 0.5（不确定性） |

### 第四步：计算损失

```python
loss = -F.logsigmoid(beta * logratios_diff)
```

- **logratios_diff > 0**：chosen 比 rejected 更受偏好 → sigmoid 接近 1 → log(1) = 0 → 损失小
- **logratios_diff < 0**：rejected 比 chosen 更受偏好 → sigmoid 接近 0 → log(小值) = 大值 → 损失大

---

## 六、直观理解

### 用"选择题"比喻

假设你有一道题的两个答案：

```
题目：什么是机器学习？

A（chosen）：机器学习是人工智能的一个分支，它使计算机能够从数据中学习和改进...
B（rejected）：机器学习就是让机器学习。
```

**问题**：哪个答案更像机器会"创造"出来的？

- 如果策略模型给 A 打分比 B 高 → 说明模型学到了"好答案"的样子
- 如果策略模型给 B 打分比 A 高 → 说明模型还在乱猜

**DPO 的目标**：让策略模型学会区分 A 和 B，而不是死记硬背。

### 用"概率比"理解

```python
logratios_diff = chosen_logratios - rejected_logratios
```

这衡量的是：**策略模型认为"chosen 比 rejected 好"的程度**。

```
logratios_diff > 0 → 策略认为 chosen 更好
logratios_diff < 0 → 策略认为 rejected 更好
```

---

## 七、为什么 DPO 有效

### 数学直觉

DPO 背后的数学洞察是：在 Bradley-Terry 偏好模型下，最优策略可以用显式公式表示，不需要训练奖励模型。

换句话说：
- PPO 需要"奖励模型打一个分数 → 根据分数调整策略"，中间多了一步
- DPO 直接优化"chosen 概率 > rejected 概率"，一步到位

### 简化流程

| PPO | DPO |
|-----|-----|
| 1. 策略生成回答 | 1. 人类标注 chosen/rejected |
| 2. 奖励模型打分 | 2. 直接计算 DPO 损失 |
| 3. PPO 损失更新 | 3. 反向传播 |
| 4. 价值损失更新 | |
| 5. KL 惩罚更新 | |

DPO 减少了中间环节，训练更简单。

---

## 八、β 参数的作用

```python
beta = 0.1  # KL 惩罚系数
```

| β 值 | 效果 |
|------|------|
| **β 较大** (如 0.5) | 策略偏离参考的惩罚更重，模型更保守 |
| **β 较小** (如 0.01) | 策略更自由，但可能偏离预训练能力太多 |

**类比**：
- β 大的老师：扣分严格，稍有不规范就扣分
- β 小的老师：更宽松，鼓励自由发挥

---

## 九、训练过程解读

```python
stats = {
    'loss': 0.23,                    # DPO 损失
    'accuracy': 0.85,                # chosen 概率 > rejected 概率的比例
    'chosen_logratios_mean': 1.2,    # 策略对 chosen 的平均偏离
    'rejected_logratios_mean': -0.3, # 策略对 rejected 的平均偏离
    'logratios_diff_mean': 1.5,      # 差值（越大越好）
}
```

| 指标 | 理想趋势 |
|------|---------|
| loss | 下降 |
| accuracy | 上升（接近 1.0） |
| chosen_logratios | 上升（更倾向 chosen） |
| rejected_logratios | 下降（更远离 rejected） |
| logratios_diff | 上升 |

---

## 十、DPO vs PPO 损失对比

### PPO 损失

```
L_total = L_policy + vf_coef × L_value + kl_coef × KL
```

- 需要策略损失 + 价值损失 + KL 惩罚
- 策略损失用 PPO clip 限制更新幅度
- 价值损失用 MSE

### DPO 损失

```
L_DPO = -log σ(β × ((log π_chosen - log π_ref_chosen)
                    - (log π_rejected - log π_ref_rejected)))
```

- 一个公式搞定
- 不需要价值网络
- 不需要在线生成

---

## 十一、一句话总结

DPO 的核心是：**让策略模型学会"chosen 比 rejected 更可能来自我"**。

```
logratios_diff = (策略对chosen的偏离) - (策略对rejected的偏离)
loss = -log(sigmoid(β × logratios_diff))
```

- logratios_diff 越大 → chosen 比 rejected 更受策略偏爱 → 损失越小
- logratios_diff 越小 → rejected 比 chosen 更受策略偏爱 → 损失越大

---

## 附录：代码实现索引

| 功能 | 文件位置 | 函数名 |
|------|---------|--------|
| log_probs 计算 | `dpo_manual.py` 第 31-68 行 | `compute_log_probs` |
| DPO 损失函数 | `dpo_manual.py` 第 75-140 行 | `compute_dpo_loss` |
| DPO 更新步骤 | `dpo_manual.py` 第 180-286 行 | `dpo_update_step` |
| 策略/参考模型加载 | `dpo_manual.py` 第 429-451 行 | `train_dpo` |

