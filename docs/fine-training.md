# 基于Phi-3.5-mini-instruct的完整训练流程

下面是一个从数据转换到最终训练的完整流程，基于Microsoft/Phi-3.5-mini-instruct和Hugging Face生态系统：

```python
import os
import json
import torch
import datasets
from datasets import load_dataset, Dataset
from peft import LoraConfig
import transformers
from trl import SFTTrainer
from transformers import (
    AutoModelForCausalLM, 
    AutoTokenizer, 
    TrainingArguments,
    BitsAndBytesConfig
)

def setup_logging():
    """设置日志"""
    import logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        level=logging.INFO,
    )

def load_model_and_tokenizer(model_name="microsoft/Phi-3.5-mini-instruct"):
    """加载模型和tokenizer"""
    
    # 模型配置
    model_kwargs = {
        "use_cache": False,
        "trust_remote_code": True,
        "attn_implementation": "flash_attention_2",
        "torch_dtype": torch.bfloat16,
        "device_map": "auto"fa
    }
    
    # 加载tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.unk_token
    tokenizer.padding_side = 'right'
    
    # 加载模型
    model = AutoModelForCausalLM.from_pretrained(model_name, **model_kwargs)
    
    return model, tokenizer

def create_sample_alpaca_data():
    """创建示例Alpaca格式数据"""
    
    alpaca_data = [
        {
            "instruction": "写一个Python函数计算斐波那契数列",
            "input": "",
            "output": "def fibonacci(n):\n    if n <= 1:\n        return n\n    else:\n        return fibonacci(n-1) + fibonacci(n-2)"
        },
        {
            "instruction": "将以下英文翻译成中文",
            "input": "The weather is really nice today, let's go for a walk.",
            "output": "今天天气真好，我们出去散步吧。"
        },
        {
            "instruction": "解释机器学习的基本概念",
            "input": "",
            "output": "机器学习是人工智能的一个分支，它使计算机能够在没有明确编程的情况下学习和改进。"
        },
        {
            "instruction": "写一封求职信",
            "input": "申请软件工程师职位，有3年Python经验",
            "output": "尊敬的招聘经理：\n\n我写信申请贵公司的软件工程师职位...\n\n此致\n敬礼"
        },
        {
            "instruction": "总结以下文章的主要内容",
            "input": "人工智能正在改变世界。从医疗诊断到自动驾驶，AI技术正在各个领域产生深远影响...",
            "output": "文章主要讨论了人工智能技术在各行各业的广泛应用和深远影响。"
        }
    ]
    
    # 保存为JSON文件
    os.makedirs("./data", exist_ok=True)
    with open("./data/alpaca_sample.json", "w", encoding="utf-8") as f:
        json.dump(alpaca_data, f, ensure_ascii=False, indent=2)
    
    return "./data/alpaca_sample.json"

def convert_alpaca_to_messages_format(data_path):
    """将Alpaca格式转换为消息格式"""
    
    # 加载数据
    with open(data_path, "r", encoding="utf-8") as f:
        alpaca_data = json.load(f)
    
    messages_data = []
    
    for example in alpaca_data:
        # 构建用户消息
        user_content = example["instruction"]
        if example.get("input", "").strip():
            user_content += "\n" + example["input"]
        
        # 构建消息格式
        messages = [
            {"role": "user", "content": user_content},
            {"role": "assistant", "content": example["output"]}
        ]
        
        messages_data.append({"messages": messages})
    
    return messages_data

def prepare_dataset(tokenizer, data_path=None, use_sample_data=True):
    """准备训练数据集"""
    
    if use_sample_data:
        # 使用示例数据
        data_path = create_sample_alpaca_data()
    
    # 转换数据格式
    messages_data = convert_alpaca_to_messages_format(data_path)
    raw_dataset = Dataset.from_list(messages_data)
    
    # 应用聊天模板
    def apply_chat_template(example):
        example["text"] = tokenizer.apply_chat_template(
            example["messages"],
            tokenize=False,
            add_generation_prompt=False
        )
        return example
    
    # 处理数据集
    processed_dataset = raw_dataset.map(
        apply_chat_template,
        remove_columns=raw_dataset.column_names
    )
    
    # 分割训练集和验证集
    if len(processed_dataset) > 1:
        dataset = processed_dataset.train_test_split(test_size=0.2, seed=42)
        train_dataset = dataset["train"]
        eval_dataset = dataset["test"]
    else:
        train_dataset = processed_dataset
        eval_dataset = None
    
    return train_dataset, eval_dataset

def setup_training_config():
    """设置训练配置"""
    
    # 训练参数
    training_config = {
        "output_dir": "./phi3-5-mini-finetuned",
        "overwrite_output_dir": True,
        "per_device_train_batch_size": 2,
        "per_device_eval_batch_size": 2,
        "gradient_accumulation_steps": 4,
        "learning_rate": 5.0e-05,
        "num_train_epochs": 3,
        "max_steps": -1,
        "lr_scheduler_type": "cosine",
        "warmup_ratio": 0.1,
        "logging_steps": 10,
        "save_steps": 100,
        "eval_steps": 100,
        "save_total_limit": 2,
        "evaluation_strategy": "steps",
        "bf16": True,
        "remove_unused_columns": False,
        "gradient_checkpointing": True,
        "gradient_checkpointing_kwargs": {"use_reentrant": False},
        "report_to": "none",  # 禁用wandb等报告
    }
    
    # LoRA配置
    peft_config = {
        "r": 16,
        "lora_alpha": 32,
        "lora_dropout": 0.05,
        "bias": "none",
        "task_type": "CAUSAL_LM",
        "target_modules": "all-linear",
    }
    
    return TrainingArguments(**training_config), LoraConfig(**peft_config)

def train_model():
    """完整的训练流程"""
    
    print("🚀 开始Phi-3.5-mini-instruct微调流程...")
    
    # 1. 设置日志
    setup_logging()
    
    # 2. 加载模型和tokenizer
    print("📥 加载模型和tokenizer...")
    model, tokenizer = load_model_and_tokenizer()
    
    # 3. 准备数据
    print("📊 准备训练数据...")
    train_dataset, eval_dataset = prepare_dataset(
        tokenizer, 
        use_sample_data=True  # 设置为False并使用data_path参数来使用你自己的数据
    )
    
    print(f"训练样本数: {len(train_dataset)}")
    if eval_dataset:
        print(f"验证样本数: {len(eval_dataset)}")
    
    # 4. 设置训练配置
    print("⚙️ 配置训练参数...")
    training_args, peft_config = setup_training_config()
    
    # 5. 创建训练器
    print("🎯 创建SFTTrainer...")
    trainer = SFTTrainer(
        model=model,
        args=training_args,
        peft_config=peft_config,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        dataset_text_field="text",
        tokenizer=tokenizer,
        max_seq_length=2048,
        packing=True,
    )
    
    # 6. 开始训练
    print("🔥 开始训练...")
    train_result = trainer.train()
    
    # 7. 保存结果
    print("💾 保存训练结果...")
    metrics = train_result.metrics
    trainer.log_metrics("train", metrics)
    trainer.save_metrics("train", metrics)
    trainer.save_state()
    
    # 8. 保存模型
    print("💿 保存微调后的模型...")
    trainer.save_model(training_args.output_dir)
    tokenizer.save_pretrained(training_args.output_dir)
    
    print(f"✅ 训练完成！模型保存在: {training_args.output_dir}")
    
    return trainer

def test_trained_model(model_path, test_questions):
    """测试训练后的模型"""
    
    print("\n🧪 测试训练后的模型...")
    
    # 加载训练后的模型和tokenizer
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True
    )
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    
    for question in test_questions:
        # 构建消息
        messages = [
            {"role": "user", "content": question}
        ]
        
        # 应用聊天模板
        text = tokenizer.apply_chat_template(
            messages, 
            tokenize=False, 
            add_generation_prompt=True
        )
        
        # 生成回答
        inputs = tokenizer(text, return_tensors="pt").to(model.device)
        outputs = model.generate(
            **inputs,
            max_new_tokens=256,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
        )
        
        # 解码输出
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        print(f"\n🤔 问题: {question}")
        print(f"🤖 回答: {response[len(text):]}")
        print("-" * 50)

if __name__ == "__main__":
    # 运行完整训练流程
    trainer = train_model()
    
    # 测试训练后的模型
    test_questions = [
        "写一个Python函数计算阶乘",
        "将'Sample text for translation'翻译成中文",
        "解释深度学习的基本概念"
    ]
    
    test_trained_model("./phi3-5-mini-finetuned", test_questions)
```

## 📁 项目文件结构

```
phi3-training/
├── train_phi3.py              # 主训练脚本
├── data/
│   ├── alpaca_sample.json     # 示例数据
│   └── your_custom_data.json  # 你的自定义数据
├── phi3-5-mini-finetuned/     # 训练输出目录
│   ├── adapter_model.safetensors
│   ├── adapter_config.json
│   └── tokenizer_config.json
└── requirements.txt
```

## 📋 环境要求

创建`requirements.txt`文件：

```txt
torch>=2.0.0
transformers>=4.37.0
datasets>=2.14.0
accelerate>=0.24.0
peft>=0.7.0
trl>=0.7.0
bitsandbytes>=0.41.0
flash-attn>=2.0.0
```

安装依赖：
```bash
pip install -r requirements.txt
```

## 🎯 使用你自己的数据

要使用你自己的数据，只需：

1. **准备Alpaca格式数据**：
```python
your_data = [
    {
        "instruction": "你的指令",
        "input": "可选输入", 
        "output": "期望输出"
    },
    # 更多数据...
]
```

2. **修改数据加载**：
```python
# 在train_model函数中修改这一行：
train_dataset, eval_dataset = prepare_dataset(
    tokenizer, 
    data_path="path/to/your/data.json",  # 你的数据路径
    use_sample_data=False  # 不使用示例数据
)
```

## ⚡ 训练优化建议

1. **显存优化**：
   - 减小`per_device_train_batch_size`
   - 增加`gradient_accumulation_steps`
   - 使用`4-bit`量化（需要修改模型加载）

2. **质量优化**：
   - 增加训练数据量和质量
   - 调整学习率（通常在1e-5到5e-5之间）
   - 增加训练轮数

3. **速度优化**：
   - 使用更强大的GPU
   - 启用混合精度训练（已启用bf16）

这个完整流程涵盖了从数据准备到模型测试的所有步骤，你可以直接使用或根据需要进行修改。