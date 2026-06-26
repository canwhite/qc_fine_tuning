# DPO (Direct Preference Optimization) 详解（小白版）

> 本文档讲解 `dpo_preference.py` 的逻辑，适合初学者阅读。

---

## 一、先理解核心概念（用类比）

想象你在教小孩选水果：

| 方法 | 类比 | 过程 |
|------|------|------|
| **PPO** | 让小孩自己挑水果 → 告诉他"好/坏" → 调整 | 需要试错 |
| **DPO** | 直接告诉小孩"苹果好，烂橙子不好" → 学会选择 | 不需要试错 |

**DPO 的核心思想**：
- 直接给模型"好回答"和"坏回答"的对比
- 让模型学会：**好回答的概率要高，坏回答的概率要低**
- 不需要 Reward Model，不需要在线生成

---

## 二、文件结构（两大块）

```
dpo_preference.py
│
├── 第一部分：PreferenceDataset（偏好数据集）
│   └── (prompt, chosen, rejected) 三元组
│
└── 第二部分：main()（主训练流程）
    ├── [1/6] 加载配置
    ├── [2/6] 加载 Tokenizer
    ├── [3/6] 加载模型（Policy + Reference）
    ├── [4/6] 初始化组件
    ├── [5/6] 训练 ← 核心！
    └── [6/6] 测试模型
```

---

## 三、逐块讲解

### 第一部分：偏好数据集

#### 数据格式

```python
PREFERENCE_DATA = [
    {
        "prompt": "什么是机器学习？",
        "chosen": "机器学习是人工智能的一个分支...",    # 好回答
        "rejected": "机器学习就是让机器学习。",         # 坏回答
    },
    {
        "prompt": "如何学习编程？",
        "chosen": "建议从以下步骤开始...",              # 好回答
        "rejected": "随便学学就行了。",                 # 坏回答
    },
    ...
]
```

**理解**：

| 字段 | 含义 |
|------|------|
| `prompt` | 问题 |
| `chosen` | 人类选择的"好回答" |
| `rejected` | 人类拒绝的"坏回答" |

---

### 第二部分：main() 主流程

#### 步骤 1：配置参数

```python
config = DPOConfig(
    model_name_or_path="Qwen/Qwen2.5-0.5B-Instruct",
    learning_rate=5e-7,      # 学习率（DPO 用很小的值）
    beta=0.1,                # KL 惩罚系数（关键参数！）
    ...
)
```

**关键参数解释**：

| 参数 | 含义 | 作用 |
|------|------|------|
| `beta` | KL 惩罚系数 | 控制模型偏离参考模型的程度 |
| `learning_rate` | 学习率 | DPO 通常用很小的值（5e-7） |

**beta 的作用**：
```
beta 越大 → 模型越保守 → 更接近原始模型
beta 越小 → 模型越激进 → 更倾向于偏好数据

推荐值：0.1 ~ 0.5
```

---

#### 步骤 2-3：加载 Tokenizer 和模型

```python
# Policy 模型（会更新）
model = AutoModelForCausalLM.from_pretrained(...)

# Reference 模型（冻结）
ref_model = AutoModelForCausalLM.from_pretrained(...)
ref_model.eval()  # 冻结参数
```

**DPO vs PPO 模型架构区别**：

| 方法 | 需要的模型 |
|------|-----------|
| PPO | Policy + ValueHead + Reference + **Reward Model** |
| DPO | Policy + Reference（**不需要 ValueHead 和 Reward Model**）|

---

#### 步骤 4：初始化组件

```python
dataset = PreferenceDataset(PREFERENCE_DATA)
dpo_trainer = DPOTrainer(
    model=model,
    ref_model=ref_model,
    args=config,
    train_dataset=dataset,
    tokenizer=tokenizer,
)
```

---

#### 步骤 5：训练（最核心！）

**DPO 训练流程**：

```
┌─────────────────────────────────────────────────────────┐
│  输入：(prompt, chosen, rejected)                        │
│                                                          │
│  prompt: "什么是机器学习？"                              │
│  chosen: "机器学习是人工智能的一个分支..."               │
│  rejected: "机器学习就是让机器学习。"                    │
└─────────────────────────────────────────────────────────┘
                           ↓
┌─────────────────────────────────────────────────────────┐
│  Step 1: 计算 chosen 的对数概率                          │
│                                                          │
│  log_prob_chosen = model.log_prob(prompt + chosen)      │
│                                                          │
│  Policy 模型对"好回答"的打分                             │
└─────────────────────────────────────────────────────────┘
                           ↓
┌─────────────────────────────────────────────────────────┐
│  Step 2: 计算 rejected 的对数概率                        │
│                                                          │
│  log_prob_rejected = model.log_prob(prompt + rejected)  │
│                                                          │
│  Policy 模型对"坏回答"的打分                             │
└─────────────────────────────────────────────────────────┘
                           ↓
┌─────────────────────────────────────────────────────────┐
│  Step 3: 计算 DPO Loss                                   │
│                                                          │
│  目标：让 chosen 的概率 > rejected 的概率                │
│                                                          │
│  Loss = -log(sigmoid(beta × (log_prob_chosen             │
│                            - log_prob_rejected)))        │
│                                                          │
│  解释：                                                  │
│  - log_prob_chosen > log_prob_rejected → Loss 小（好）   │
│  - log_prob_chosen < log_prob_rejected → Loss 大（惩罚） │
└─────────────────────────────────────────────────────────┘
                           ↓
┌─────────────────────────────────────────────────────────┐
│  Step 4: 反向传播                                        │
│                                                          │
│  更新 Policy 模型，让它更倾向于 chosen                   │
│  Reference 模型保持不变                                  │
└─────────────────────────────────────────────────────────┘
```

---

## 四、DPO Loss 数学解释

### 损失函数

```
L_DPO = -log(sigmoid(β × (log π_chosen - log π_rejected)))
```

### 分步理解

```python
import torch
import torch.nn.functional as F

# 假设模型输出的对数概率
log_prob_chosen = -0.5    # 好回答的概率（取 log 后）
log_prob_rejected = -2.0  # 坏回答的概率

# 差距
diff = log_prob_chosen - log_prob_rejected  # = 1.5

# 乘 beta
beta = 0.1
scaled_diff = beta * diff  # = 0.15

# Sigmoid
sigmoid = torch.sigmoid(torch.tensor(scaled_diff))  # ≈ 0.54

# Loss
loss = -torch.log(sigmoid)  # ≈ 0.62
```

### 不同情况分析

| 情况 | diff | sigmoid | loss | 含义 |
|------|------|---------|------|------|
| chosen 概率 >> rejected | +2.0 | 0.88 | 0.13 | 很好，loss 小 |
| chosen 概率 > rejected | +0.5 | 0.62 | 0.48 | 还可以 |
| chosen 概率 = rejected | 0.0 | 0.50 | 0.69 | 一般，需要改进 |
| chosen 概率 < rejected | -0.5 | 0.38 | 0.97 | 不好，需要调整 |
| chosen 概率 << rejected | -2.0 | 0.12 | 2.12 | 很差，惩罚大 |

---

## 五、DPO vs PPO vs 蒸馏 对比

```
┌─────────────────────────────────────────────────────────┐
│                        PPO                               │
├─────────────────────────────────────────────────────────┤
│  prompt ──→ 模型生成 ──→ Reward Model 打分 ──→ 调整     │
│                                                          │
│  需要在线生成，需要探索，样本效率低                       │
│  需要：Policy + ValueHead + Reference + Reward Model     │
└─────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────┐
│                        DPO                               │
├─────────────────────────────────────────────────────────┤
│  (prompt, chosen, rejected) ──→ 直接学习偏好             │
│                                                          │
│  离线训练，不需要生成，不需要 Reward Model                │
│  需要：Policy + Reference                                │
└─────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────┐
│                        蒸馏                              │
├─────────────────────────────────────────────────────────┤
│  教师 logits ──→ 学生学习软标签                          │
│                                                          │
│  模仿教师的"思考过程"                                    │
│  需要：Teacher + Student                                 │
└─────────────────────────────────────────────────────────┘
```

---

## 六、关键概念总结

| 概念 | 解释 |
|------|------|
| **偏好数据** | (prompt, chosen, rejected) 三元组 |
| **chosen** | 人类选择的好回答 |
| **rejected** | 人类拒绝的坏回答 |
| **beta** | KL 惩罚系数，控制偏离程度 |
| **DPO Loss** | 让 chosen 概率 > rejected 概率 |
| **Reference 模型** | 冻结的原始模型，防止偏离太远 |

---

## 七、一句话总结

```
DPO = 直接从偏好数据学习
    = 不需要 Reward Model
    = 不需要在线生成
    = 让模型更倾向于"好回答"而非"坏回答"

本质：把 RLHF 简化成分类问题
```

---

## 八、参考资料

- [DPO 论文 (Direct Preference Optimization, 2023)](https://arxiv.org/abs/2305.18290)
- [TRL DPO 文档](https://huggingface.co/docs/trl/main/en/dpo_trainer)