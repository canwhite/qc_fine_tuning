#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Qwen2.5-0.5B-Instruct 模型微调脚本 (修复版本)
解决SFTTrainer数据格式问题
"""

import json
import logging
import os
import time
import warnings

warnings.filterwarnings("ignore")

import torch
from datasets import Dataset
from peft import LoraConfig #parameter efficient
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments # some tools for sft
from trl import SFTTrainer # trainer

# 设置M2 Air优化环境变量
os.environ["PYTORCH_MPS_HIGH_WATERMARK_RATIO"] = "0.0"
os.environ["OMP_NUM_THREADS"] = "1"

# 设置随机种子
# 训练中涉及随机性的部分，如数据分割、参数初始化等，
# 设置固定的随机种子可以确保每次运行得到相同的结果，方便调试和复现。
torch.random.manual_seed(42)

logger = logging.getLogger(__name__)


def setup_logging():
    """设置日志"""
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        level=logging.INFO,
    )


# loadmodel
def load_qwen_model(model_name="Qwen/Qwen2.5-0.5B-Instruct"):
    """加载Qwen2.5模型"""
    logger.info(f"开始加载Qwen2.5模型: {model_name}")

    model_kwargs = {
        "torch_dtype": torch.float16,
        "device_map": None,
        "trust_remote_code": True,
        "use_cache": False,
        "low_cpu_mem_usage": True,
    }

    try:
        # 加载tokenizer
        tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        # 分词器：
        # 玩具火车的“节数”就是一句话被切成的 token 数量，
        # 短的那列火车后面挂的“空车厢”就是 pad_token（如 [PAD]
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        # 在序列的右侧添加padding（默认是左侧）
        tokenizer.padding_side = "right"

        # 加载模型，微调一般是用这个AutoModelForCausalLM
        # 另外常用的AutoModel没任务头的，
        # 以及AutoModelForSequenceClassification，加了一个线性分类头，这个ppo和dpo的时候用
        model = AutoModelForCausalLM.from_pretrained(model_name, **model_kwargs)

        # 移动到MPS
        # 这里的backends指的是GPU后端，MPS是苹果的GPU后端，类似于CUDA（NVIDIA）和ROCm（AMD）
        if torch.backends.mps.is_available():
            model = model.to("mps")
            logger.info("✅ 模型已移动到MPS")
        else:
            logger.info("💻 使用CPU")

        # 模型参数统计，就是前向传播时需要计算的参数数量，通常以百万（M）或十亿（B）为单位  
        total_params = sum(p.numel() for p in model.parameters())
        logger.info(f"模型总参数: {total_params:,}")

        # 最后返回model和tokenizer，后续训练和生成都需要用到它们
        return model, tokenizer

    except Exception as e:
        logger.error(f"❌ 模型加载失败: {e}")
        raise


# 准备数据集的基本函数
def create_training_data():
    """创建训练数据 - 使用更简单的格式"""
    #chatML格式的训练数据，直接用text字段存储完整的对话文本，包含用户和助手的角色标记
    """
    问答对儿：                                                                                                                                  
        <|im_start|>user
        问题                                                                                                                             
        <|im_end|>
        <|im_start|>assistant                                                                                                            
        回答                                                      
        <|im_end|>
    """
    training_data = [
        # 简单的对话格式
        {
            "text": '<|im_start|>user\n用Python写一个计算圆面积的函数<|im_end|>\n<|im_start|>assistant\nimport math\n\ndef calculate_circle_area(radius):\n    if radius < 0:\n        raise ValueError("半径不能为负数")\n    return math.pi * radius ** 2\n\narea = calculate_circle_area(5)\nprint(f"面积: {area}")<|im_end|>'
        },
        {
            "text": "<|im_start|>user\n解释什么是量子计算<|im_end|>\n<|im_start|>assistant\n量子计算是一种基于量子力学原理的计算方式。\n\n核心特点：\n1. 量子比特：可以同时处于0和1的叠加态\n2. 量子纠缠：粒子间的特殊关联\n3. 量子并行：天然支持大规模并行计算\n\n应用领域：密码学、药物设计、人工智能等。<|im_end|>"
        },
        {
            "text": "<|im_start|>user\n将'人工智能改变世界'翻译成英文<|im_end|>\n<|im_start|>assistant\nArtificial intelligence is changing the world.<|im_end|>"
        },
        {
            "text": "<|im_start|>user\n写一个冒泡排序算法<|im_end|>\n<|im_start|>assistant\ndef bubble_sort(arr):\n    n = len(arr)\n    for i in range(n):\n        for j in range(0, n-i-1):\n            if arr[j] > arr[j+1]:\n                arr[j], arr[j+1] = arr[j+1], arr[j]\n    return arr\n\n# 测试\nprint(bubble_sort([64, 34, 25, 12, 22, 11, 90]))<|im_end|>"
        },
        {
            "text": "<|im_start|>user\n什么是机器学习？<|im_end|>\n<|im_start|>assistant\n机器学习是人工智能的一个分支，让计算机能够从数据中学习规律，而无需明确编程。\n\n主要类型：\n1. 监督学习：从标注数据学习\n2. 无监督学习：从无标注数据发现模式\n3. 强化学习：通过奖励机制学习\n\n应用：图像识别、自然语言处理、推荐系统等。<|im_end|>"
        },
        {
            "text": "<|im_start|>user\n计算：123 + 456 = ?<|im_end|>\n<|im_start|>assistant\n123 + 456 = 579<|im_end|>"
        },
        {
            "text": "<|im_start|>user\n写一个简单的Flask应用<|im_end|>\n<|im_start|>assistant\nfrom flask import Flask\n\napp = Flask(__name__)\n\n@app.route('/')\ndef hello():\n    return 'Hello, World!'\n\nif __name__ == '__main__':\n    app.run(debug=True)<|im_end|>"
        },
    ]

    # 保存数据
    os.makedirs("./data", exist_ok=True)
    data_path = "./data/training_data.json"
    with open(data_path, "w", encoding="utf-8") as f:
        """
        # 序列化到字符串
        json_str = json.dumps(training_data, ensure_ascii=False, indent=2)                                                               
                                                                    
        # 字符串转回Python对象                                                                                                           
        data = json.loads(json_str)
        """
        #dumps是转成字符串，这里是？dump直接写入文件，ensure_ascii=False保持中文，indent=2格式化输出
        json.dump(training_data, f, ensure_ascii=False, indent=2)

    logger.info(f"已创建训练数据: {data_path}")
    return data_path


def prepare_dataset(tokenizer):
    """准备训练数据集 - 使用text格式"""
    # 创建训练数据
    data_path = create_training_data()

    # 加载数据
    with open(data_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    # 创建数据集
    dataset = Dataset.from_list(data)

    # 分割数据集为训练集和验证集，20%作为验证集，如果数据量太小就不分割了
    if len(dataset) > 2:
        split_dataset = dataset.train_test_split(test_size=0.2, seed=42)
        #分割好的是两个map，train和test，分别对应训练集和验证集
        train_dataset = split_dataset["train"]
        eval_dataset = split_dataset["test"]
    else:
        train_dataset = dataset
        eval_dataset = None

    logger.info(f"训练集样本数: {len(train_dataset)}")
    logger.info(f"验证集样本数: {len(eval_dataset) if eval_dataset else 0}")

    # 显示样本格式
    if len(train_dataset) > 0:
        logger.info(f"训练数据示例: {train_dataset[0]}")

    return train_dataset, eval_dataset


def create_lora_config():
    """创建LoRA配置"""
    """关键指标start★★★ LoRA核心参数 ★★★
    r= 16,         # 最关键指标，增加r值，中间秩更大，理论上能捕捉更多信息，但也更重量
    lora_alpha=32, # 这个是缩放因子，通常设置为r的倍数，output = W · x + (α/r) · ΔW · x
                   # ΔW = B·A，B 和 A 是低秩矩阵，参数量小得多， 但通过缩放 α/r 放大后，效果显著。
    """
    return LoraConfig(
        #key metrics
        r=16, 
        lora_alpha=32, 

        #others
        lora_dropout=0.1, #随机丢弃随机丢弃 10% 的 LoRA 路径，防止过拟合，和relu实现的作用很想，但不是完全一样，
        #relu是丢弃部分神经元的输出，而lora_dropout是丢弃部分LoRA路径的输出
        bias="none", # 不训练偏置参数，保持原模型的偏置不变，减少训练参数量
        task_type="CAUSAL_LM", # 任务类型，CAUSAL_LM表示因果语言模型，适用于文本生成任务
        target_modules=[ #指定把 LoRA 应用到哪些层。列出的都是 Attention 和 FFN 的投影层 
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ],
    )


def create_training_args():
    """创建训练参数"""
    output_dir = "./qwen_fixed_finetuned"

    return TrainingArguments(
        output_dir=output_dir,
        # ═════════════════════════════════════════════════════════
        # ★★★ 核心参数：直接决定训练成败
        # ═════════════════════════════════════════════════════════
        learning_rate=5e-5,  # 学习率：步子大小
        num_train_epochs=5,  # 训练轮数：走几圈
        per_device_train_batch_size=1,  # 批大小：一次喂多少
        gradient_accumulation_steps=4,  # 梯度累积：显存不够时用，将几个batch的梯度累积起来再更新一次，相当于在原有的批次上再分批
        per_device_eval_batch_size=1,  # 评估批大小

        # ═════════════════════════════════════════════════════════
        # ★★ 重要参数：影响训练稳定性
        # ═════════════════════════════════════════════════════════
        lr_scheduler_type="cosine",  # 学习率调度：cosine慢慢减小
        warmup_ratio=0.1,  # 预热：开头学习率慢慢升
        max_grad_norm=1.0,  # 梯度裁剪：防止步子太大
        # 保存和评估
        do_eval=True,
        eval_strategy="steps",
        eval_steps=50,  # 每50步评估（原来是2，太频繁）
        save_strategy="steps",
        save_steps=50,  # 每50步保存（原来是2，太频繁）
        save_total_limit=3,  # 最多保留3个checkpoint
        load_best_model_at_end=True,  # 结束时加载最优模型
        # ═════════════════════════════════════════════════════════
        # ○ 辅助参数：有默认值，按需调整
        # ═════════════════════════════════════════════════════════
        weight_decay=0.01,  # 权重衰减：防止过拟合
        # ═════════════════════════════════════════════════════════
        # ○ 默认即可：通常不需要改
        # ═════════════════════════════════════════════════════════
        optim="adamw_torch",  # 优化器
        adam_beta1=0.9,  # Adam参数
        adam_beta2=0.999,  # Adam参数
        adam_epsilon=1e-8,  # Adam参数
        # 日志
        logging_steps=10,  # 每10步打印日志
        logging_strategy="steps",
        # 其他
        overwrite_output_dir=True,
        dataloader_pin_memory=False,  # MPS优化
        remove_unused_columns=False,  # 保留text列
        seed=42,  # 随机种子
        report_to="none",  # 禁用wandb
    )


# 从最终阶段开始
def main():
    """主训练函数"""
    setup_logging()
    logger.info("开始修复版Qwen2.5-0.5B模型微调...")

    try:
        # 1. 加载模型和tokenizer
        logger.info("步骤1: 加载模型和tokenizer")
        model, tokenizer = load_qwen_model()

        # 2. 准备数据集
        logger.info("步骤2: 准备训练数据集")
        train_dataset, eval_dataset = prepare_dataset(tokenizer)

        # 3. 创建训练配置
        logger.info("步骤3: 配置训练参数")
        training_args = create_training_args()
        peft_config = create_lora_config()

        # 4. 创建训练器 - 使用formatting_func
        logger.info("步骤4: 创建训练器")

        def formatting_func(example):
            """格式化函数，返回文本"""
            return example["text"]

        trainer = SFTTrainer(
            model=model, # model
            args=training_args, #训练参数
            peft_config=peft_config, #lora参数
            train_dataset=train_dataset, # 训练数据
            eval_dataset=eval_dataset, #评估数据
            # 关键参数
            formatting_func=formatting_func, # 取值example["text"]，直接返回文本，不需要额外处理，SFTTrainer会自动调用tokenizer进行编码
            processing_class=tokenizer, #分词器
        )

        # 5. 显示训练信息
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

        logger.info(f"模型参数统计:")
        logger.info(f"  总参数: {total_params:,}")
        logger.info(f"  可训练参数: {trainable_params:,}")
        logger.info(f"  训练参数占比: {100 * trainable_params / total_params:.2f}%")
        logger.info(f"  训练样本: {len(train_dataset)}")
        logger.info(f"  验证样本: {len(eval_dataset) if eval_dataset else 0}")

        # 6. 开始训练
        logger.info("步骤5: 开始训练...")
        start_time = time.time()
        # 直接调用train函数就ok，这个确实简单
        train_result = trainer.train()

        end_time = time.time()
        logger.info(f"训练完成！总耗时: {end_time - start_time:.2f}秒")

        # 7. 训练结果
        logger.info(f"训练统计:")
        logger.info(f"  最终训练损失: {train_result.training_loss:.6f}")
        logger.info(f"  总训练步数: {train_result.global_step}")

        # 8. 保存模型
        trainer.save_model()
        tokenizer.save_pretrained(training_args.output_dir)
        logger.info(f"✅ 模型已保存到: {training_args.output_dir}")

        # 9. 评估模型
        if eval_dataset:
            logger.info("开始评估微调后的模型...")
            eval_metrics = trainer.evaluate()
            logger.info(f"验证集评估结果: {eval_metrics}")

    except Exception as e:
        logger.error(f"❌ 训练失败: {e}")
        raise


if __name__ == "__main__":
    main()
