# QC Fine Tuning

大模型微调工具集，支持 SFT、DPO、PPO、Distillation 等多种训练方法，可在 Mac M 系列芯片 (MPS) 或 GPU 上运行。

## 项目结构

```
qc_fine_tuning/
├── 核心训练脚本
│   ├── fine_tune_qwen.py       # Qwen2.5 SFT 微调
│   ├── fine_tune_phi.py        # Phi SFT 微调
│   ├── ppo.py                  # PPO + RLHF 训练
│   ├── ppo_manual.py           # PPO 手动实现 (详细注释版)
│   ├── dpo.py                  # DPO 偏好优化训练
│   ├── dpo_manual.py           # DPO 手动实现 (详细注释版)
│   └── distillation_demo.py    # 知识蒸馏示例
│
├── 模型使用脚本
│   ├── use_qwen2.5.py          # 使用微调后的 Qwen 模型
│   ├── use_qwen2.5_lora.py    # 使用 Qwen + LoRA
│   ├── use_phi_lora.py         # 使用 Phi + LoRA
│   └── use_phi_m2air.py        # M2 Air 专用 Phi
│
├── 数据目录
│   └── data/                   # 训练数据
│
└── 文档 (docs/)
    ├── SFT 相关
    │   ├── fine-training.md
    │   ├── tokenizer_and_attention.md
    │   └── training_data_format.md
    ├── DPO 相关
    │   ├── dpo_complete_guide.md
    │   ├── dpo_explanation.md
    │   ├── dpo_loss_for_beginners.md
    │   └── dpo_vs_ppo_comparison.md
    ├── PPO/RLHF 相关
    │   ├── ppo_complete_guide.md
    │   ├── ppo_rlhf_explanation.md
    │   ├── ppo_loss_for_beginners.md
    │   └── ppo_dpo_manual_implementation.md
    ├── 知识蒸馏
    │   ├── distillation_details.md
    │   └── distillation_explanation.md
    └── 其他
        ├── lora_and_low_rank_explanation.md
        ├── sft_trainer_advantages.md
        └── summary.md
```

## 支持的训练方法

| 方法 | 说明 |
|------|------|
| **SFT** | Supervised Fine-Tuning，监督微调 |
| **DPO** | Direct Preference Optimization，直接偏好优化 |
| **PPO + RLHF** | Proximal Policy Optimization + Reinforcement Learning from Human Feedback |
| **Distillation** | 知识蒸馏 |

## 支持的模型

- **Qwen2.5-0.5B-Instruct**
- **Phi-3-mini**
- 支持 LoRA 高效微调

## 环境配置

### 使用 UV (推荐)

```bash
# 直接运行，自动管理虚拟环境
uv run python fine_tune_qwen.py

# 手动管理依赖
uv add transformers peft datasets trl torch
uv run python your_script.py
```

### 依赖

- Python >= 3.10
- PyTorch 2.4.1 (MPS 优化版)
- transformers
- peft (LoRA)
- trl (SFTTrainer, DPOTrainer, PPOTrainer)
- datasets

## 快速开始

### SFT 微调

```bash
uv run python fine_tune_qwen.py
```

### DPO 训练

```bash
uv run python dpo.py
```

### PPO 训练

```bash
uv run python ppo.py
```

## 文档

详细说明请参考 `docs/` 目录下的文档：

- 新手入门：[dpo_loss_for_beginners.md](docs/dpo_loss_for_beginners.md)、[ppo_loss_for_beginners.md](docs/ppo_loss_for_beginners.md)
- 算法对比：[dpo_vs_ppo_comparison.md](docs/dpo_vs_ppo_comparison.md)
- 完整指南：[dpo_complete_guide.md](docs/dpo_complete_guide.md)、[ppo_complete_guide.md](docs/ppo_complete_guide.md)
