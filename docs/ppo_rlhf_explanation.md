# PPO + RLHF 训练详解（小白版）

> 本文档讲解 `ppo_rlhf_with_real_rm.py` 的逻辑，适合初学者阅读。

---

## 一、先理解核心概念（用类比）

想象你在训练一只小狗：

| 角色 | 类比 | 作用 |
|------|------|------|
| **Policy 模型** | 小狗 | 被训练的对象，会生成回答 |
| **Reward Model** | 主人 | 给小狗的表现打分（好/坏） |
| **ValueHead** | 小狗的预期 | 小狗觉得自己能得多少分 |
| **PPO 算法** | 训练方法 | 根据打分调整小狗的行为 |

**核心思想**：
- 小狗做出动作 → 主人打分 → 小狗记住"这样做得分高" → 下次多做
- 但不能改太多，否则小狗会"忘记"之前学的东西

---

## 二、文件结构（三大块）

```
ppo_rlhf_with_real_rm.py
│
├── 第一部分：RealRewardModel（真实打分器）
│   └── 从 HuggingFace 加载预训练的奖励模型
│
├── 第二部分：PPODataset（训练数据）
│   └── 一堆问题，让模型回答
│
└── 第三部分：main()（主训练流程）
    ├── [1/7] 加载配置
    ├── [2/7] 加载 Tokenizer
    ├── [3/7] 加载模型（带 ValueHead）
    ├── [4/7] 加载 Reward Model
    ├── [5/7] 初始化组件
    ├── [6/7] 训练循环 ← 核心！
    └── [7/7] 测试模型
```

---

## 三、Reward Model 详解

### 什么是 Reward Model？

Reward Model 是一个用人类偏好数据训练的神经网络，用于给模型的回答打分：

```
输入：问题 + 回答
输出：标量奖励分数（越高越好）
```

### 代码实现

```python
class RealRewardModel:
    def __init__(self, model_name: str):
        # 从 HuggingFace 加载预训练模型
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    def __call__(self, query: str, response: str) -> float:
        # 构造输入格式
        text = f"Human: {query}\n\nAssistant: {response}"
        
        # 模型推理
        inputs = self.tokenizer(text, return_tensors="pt")
        with torch.no_grad():
            outputs = self.model(**inputs)
            reward = outputs.logits[0, 0].item()
        
        return reward
```

### 可用的开源 Reward Model

| 模型 | 语言 | 大小 | 推荐场景 |
|------|------|------|----------|
| `OpenAssistant/reward-model-deberta-v3-large-v2` | 英文 | ~1.5B | 英文训练（推荐） |
| `OpenAssistant/reward-model-deberta-v3-base` | 英文 | ~400M | 轻量英文训练 |
| `Ray2333/reward-model-reward-model-english-chinese` | 中英文 | ~1.5B | 中文训练 |

### Reward Model 工作原理

```
┌─────────────────────────────────────────────────────────┐
│  真实 Reward Model                                       │
├─────────────────────────────────────────────────────────┤
│                                                          │
│  打分逻辑：                                              │
│    - 用人类偏好数据训练的神经网络                        │
│    - 理解语义和逻辑                                      │
│    - 真实反映"人类觉得好不好"                            │
│                                                          │
│  训练过程：                                              │
│    1. 收集人类偏好数据（好回答 vs 坏回答）               │
│    2. 用分类任务训练模型                                 │
│    3. 模型学会判断"哪个回答更好"                         │
│                                                          │
│  输入格式：                                              │
│    "Human: 什么是AI?\n\nAssistant: AI是人工智能..."      │
│                                                          │
│  输出：                                                  │
│    奖励分数（如 1.8）                                    │
│                                                          │
└─────────────────────────────────────────────────────────┘
```

---

## 四、main() 主流程详解

### 步骤 1：配置参数

```python
config = PPOConfig(
    model_name="Qwen/Qwen2.5-0.5B-Instruct",  # Policy 模型
    learning_rate=1e-5,      # 学习率（PPO 用小值）
    batch_size=2,            # 批大小
    ppo_epochs=4,            # 每个 batch 的 PPO 更新次数
    init_kl_coef=0.2,        # KL 约束系数
)
```

**关键参数解释**：
- `kl_coef`：约束模型不要偏离原始模型太远

---

### 步骤 2-3：加载 Tokenizer 和模型

```python
# Policy 模型：会更新（小狗）
model = AutoModelForCausalLMWithValueHead.from_pretrained(...)

# Reference 模型：冻结不动（作为参照物）
ref_model = create_reference_model(model)
```

**理解**：
- `model`：被训练的模型，会不断更新
- `ref_model`：原始模型的副本，冻结不动，用来计算 KL 散度

---

### 步骤 4：加载 Reward Model

```python
reward_model = RealRewardModel(
    model_name="OpenAssistant/reward-model-deberta-v3-large-v2",
    device=device,
)
```

---

### 步骤 5：初始化组件

```python
dataset = PPODataset(TRAINING_QUERIES)
dataloader = DataLoader(dataset, batch_size=config.batch_size)
ppo_trainer = PPOTrainer(
    config=config,
    model=model,
    ref_model=ref_model,
    tokenizer=tokenizer,
)
```

---

### 步骤 6：训练循环（最核心！）

每个 batch 的训练流程：

```
┌─────────────────────────────────────────────────────────┐
│  Step 1: 编码问题                                        │
│  "What is AI?" → [101, 102, 103, ...] (token IDs)       │
└─────────────────────────────────────────────────────────┘
                           ↓
┌─────────────────────────────────────────────────────────┐
│  Step 2: 模型生成回答                                    │
│  模型生成："AI is a branch of computer science..."       │
└─────────────────────────────────────────────────────────┘
                           ↓
┌─────────────────────────────────────────────────────────┐
│  Step 3: 解码回答                                        │
│  [201, 202, ...] → "AI is a branch of computer science" │
└─────────────────────────────────────────────────────────┘
                           ↓
┌─────────────────────────────────────────────────────────┐
│  Step 4: 计算奖励                                        │
│  Reward Model 推理 → 1.8 分                              │
└─────────────────────────────────────────────────────────┘
                           ↓
┌─────────────────────────────────────────────────────────┐
│  Step 5: PPO 训练步骤（核心！）                          │
│                                                          │
│  输入：问题 tokens + 回答 tokens + 奖励分数              │
│                                                          │
│  内部计算：                                              │
│  1. ValueHead 预测"预期分数"（比如 1.5）                 │
│  2. Advantage = 实际奖励 - 预期 = 1.8 - 1.5 = 0.3       │
│  3. Advantage > 0 → 这个回答比预期好！增加概率           │
│  4. 同时计算 KL 散度，防止偏离太远                       │
│  5. 更新模型参数                                         │
└─────────────────────────────────────────────────────────┘
```

**代码对应**：

```python
# Step 1: 编码
query_tensors = [tokenizer(q)["input_ids"] for q in queries]

# Step 2: 生成
response_tensors = ppo_trainer.generate(query_tensors, ...)

# Step 3: 解码
responses = [tokenizer.decode(r) for r in response_tensors]

# Step 4: 计算奖励
rewards = [reward_model(q, r) for q, r in zip(queries, responses)]

# Step 5: PPO 训练（核心！）
stats = ppo_trainer.step(queries, responses, rewards)
```

---

### 步骤 7：测试训练后的模型

```python
model.eval()
with torch.no_grad():
    for query in test_queries:
        response = model.generate(query)
        reward = reward_model(query, response)
        print(f"Q: {query}\nA: {response}\nReward: {reward}")
```

---

## 五、ValueHead 的作用（关键理解）

```
┌────────────────────────────────────────────────────────┐
│  为什么需要 ValueHead？                                 │
│                                                         │
│  没有 ValueHead：                                       │
│    直接用奖励训练 → 方差大，训练不稳定                   │
│                                                         │
│  有 ValueHead：                                         │
│    advantage = 实际奖励 - 预期奖励                      │
│    - advantage > 0：比预期好 → 增加概率                 │
│    - advantage < 0：比预期差 → 降低概率                 │
│                                                         │
│  类比：                                                 │
│    - 考试考了 80 分                                     │
│    - 没有预期：80 分是好是坏？不知道                     │
│    - 有预期（预期 60 分）：80 > 60，表现好！             │
└────────────────────────────────────────────────────────┘
```

---

## 六、总结流程图

```
问题 ──→ 模型生成回答 ──→ Reward Model 打分
                               ↓
                          PPO 训练
                               ↓
                     ┌─────────────────┐
                     │ ValueHead 预期  │
                     │ 实际奖励 - 预期  │
                     │ = Advantage     │
                     └─────────────────┘
                               ↓
                     根据 Advantage 更新模型
                               ↓
                     KL 约束：不能偏离太远
                               ↓
                          模型变好！
```

---

## 七、运行方式

```bash
uv run python ppo_rlhf_with_real_rm.py
```

**注意**：首次运行会从 HuggingFace 下载 Reward Model（约 1.5GB）。

---

## 八、相关概念补充

### KL 散度（KL Divergence）

- 衡量两个概率分布的"距离"
- 在 PPO 中用来约束模型不要偏离原始模型太远

### PPO vs DPO

| 方法 | 特点 | 适用场景 |
|------|------|----------|
| PPO | 需要在线生成 + Reward Model | 有好的 Reward Model |
| DPO | 直接用偏好数据训练 | 有成对偏好数据 |

---

## 九、参考资料

- [TRL 官方文档](https://huggingface.co/docs/trl/)
- [PPO 论文](https://arxiv.org/abs/1707.06347)
- [RLHF 综述](https://arxiv.org/abs/2203.02155)
- [OpenAssistant Reward Model](https://huggingface.co/OpenAssistant/reward-model-deberta-v3-large-v2)