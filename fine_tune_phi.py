# ==================== 标准库 ====================
import json  # JSON数据处理库，用于配置文件和数据序列化
import logging
import os  # 操作系统接口，用于文件路径操作和环境变量获取


# ==================== AI训练相关核心包 ====================
import torch  # PyTorch深度学习框架，用于张量计算和神经网络训练

# 基础用法：dataset = load_dataset("csv", data_files="data.csv")
# 主要作用：统一数据集格式，支持流式处理和高效存储
from datasets import Dataset  # 加载预置数据集和创建自定义数据集

# ==================== 高效微调相关包 ====================

# 基础用法：model = AutoModel.from_pretrained("bert-base-uncased")
# 主要作用：加载和训练Transformer架构的NLP模型
from transformers import AutoModelForCausalLM  # 自动选择因果语言模型（如GPT系列）
from transformers import AutoTokenizer  # 自动选择对应的分词器
from transformers import TrainingArguments  # 训练参数配置类
from peft import LoraConfig  # 参数高效微调库的LoRA配置
from trl import SFTTrainer  # 监督微调训练器，专为语言模型优化


logger = logging.getLogger(__name__)


def setup_logging():
    """设置日志"""
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        level=logging.INFO,
    )


def load_model_and_tokenizer(model_name="microsoft/Phi-3.5-mini-instruct"):
    """加载模型和tokenizer"""

    # 模型配置参数 - M2 Air超内存优化配置
    model_kwargs = {
        "use_cache": False,  # 训练时关闭KV缓存，节省内存；推理时可设为True加速
        "trust_remote_code": True,  # 信任模型的远程代码，某些自定义模型架构需要
        "torch_dtype": torch.float16,  # 使用float16精度，M2芯片支持
        "device_map": None,  # 完全不使用device_map，避免内存碎片
        "attn_implementation": "eager",  # 使用eager attention避免flash attention警告
        "low_cpu_mem_usage": True,  # M2 Air专用：减少CPU内存使用
    }

    # 加载tokenizer（文本预处理工具）
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.unk_token  # 如果没有填充token，用未知token代替
    tokenizer.padding_side = "right"  # 在序列右侧进行填充，符合训练惯例

    # 加载预训练的语言模型
    model = AutoModelForCausalLM.from_pretrained(model_name, **model_kwargs)

    # M2芯片：手动移动模型到MPS设备
    if torch.backends.mps.is_available():
        model = model.to("mps")
        logger.info("模型已移动到MPS设备（M2芯片）")
    else:
        logger.info("使用CPU训练")

    return model, tokenizer


def create_sample_alpaca_data():
    """创建示例Alpaca格式数据"""

    alpaca_data = [
        {
            "instruction": "写一个Python函数计算斐波那契数列",
            "input": "",
            "output": "def fibonacci(n):\n    if n <= 1:\n        return n\n    else:\n        return fibonacci(n-1) + fibonacci(n-2)",
        },
        {
            "instruction": "将以下英文翻译成中文",
            "input": "The weather is really nice today, let's go for a walk.",
            "output": "今天天气真好，我们出去散步吧。",
        },
        {
            "instruction": "解释机器学习的基本概念",
            "input": "",
            "output": "机器学习是人工智能的一个分支，它使计算机能够在没有明确编程的情况下学习和改进。",
        },
        {
            "instruction": "写一封求职信",
            "input": "申请软件工程师职位，有3年Python经验",
            "output": "尊敬的招聘经理：\n\n我写信申请贵公司的软件工程师职位...\n\n此致\n敬礼",
        },
        {
            "instruction": "总结以下文章的主要内容",
            "input": "人工智能正在改变世界。从医疗诊断到自动驾驶，AI技术正在各个领域产生深远影响...",
            "output": "文章主要讨论了人工智能技术在各行各业的广泛应用和深远影响。",
        },
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
            {"role": "assistant", "content": example["output"]},
        ]

        messages_data.append({"messages": messages})

    return messages_data


def prepare_dataset(tokenizer, data_path=None, use_sample_data=True):
    """准备训练数据集"""

    # 步骤1：决定使用什么数据
    if use_sample_data:
        # 如果use_sample_data=True，使用内置的示例数据
        data_path = create_sample_alpaca_data()  # 创建5个样本的JSON文件

    # 步骤2：转换数据格式
    # 从Alpaca格式 {"instruction": "...", "input": "...", "output": "..."}
    # 转换为聊天格式 [{"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}]
    messages_data = convert_alpaca_to_messages_format(data_path)

    # 步骤3：创建Hugging Face数据集对象
    raw_dataset = Dataset.from_list(messages_data)  # 把Python列表转为HF数据集格式

    # 步骤4：定义内部函数 - 应用聊天模板
    def apply_chat_template(example):
        example["text"] = tokenizer.apply_chat_template(
            example["messages"],  # 输入：聊天消息列表
            tokenize=False,  # 处理：不进行分词，返回字符串
            add_generation_prompt=False,  # 返回：不添加生成提示符
        )
        return example

    # 步骤5：批量处理数据集
    processed_dataset = raw_dataset.map(
        apply_chat_template,  # 对每个样本应用这个函数
        remove_columns=raw_dataset.column_names,  # 删除原来的"messages"列，只保留"text"列
    )
    logger.info(f"process data:{processed_dataset}")

    # 步骤6：分割数据集
    if len(processed_dataset) > 1:  # 如果数据多于1条
        dataset = processed_dataset.train_test_split(
            test_size=0.2, seed=42
        )  # 80%训练，20%验证
        train_dataset = dataset["train"]  # 训练集
        eval_dataset = dataset["test"]  # 验证集
    else:  # 如果只有1条数据
        train_dataset = processed_dataset  # 全部作为训练集
        eval_dataset = None  # 没有验证集

    return train_dataset, eval_dataset  # 返回训练集和验证集


def main():
    setup_logging()

    # 加载模型和数据
    model, tokenizer = load_model_and_tokenizer()
    data_path = create_sample_alpaca_data()
    train_dataset, eval_dataset = prepare_dataset(tokenizer, data_path)

    # 训练配置 - M2芯片优化
    training_config = {
        # M2芯片不需要fp16，因为MPS有自己的优化
        # "fp16": True,  # 注释掉，避免MPS设备冲突
        # 基础训练配置
        "do_eval": True,  # 启用验证集评估
        "learning_rate": 5.0e-06,  # 学习率：较小的学习率适合微调
        "log_level": "info",  # 日志级别
        "logging_steps": 1,  # 每步记录日志，便于调试
        # 调度器和日志策略
        "logging_strategy": "steps",  # 按步骤记录日志
        "lr_scheduler_type": "cosine",  # 余弦学习率调度
        # 训练轮数和步数
        "num_train_epochs": 3,  # 训练3个epoch，充分学习小数据集
        "max_steps": -1,  # 不限制最大步数，按epoch训练
        # 输出和保存配置
        "output_dir": "./phi_checkpoint",  # 模型保存目录
        "overwrite_output_dir": True,  # 覆盖之前的输出目录
        # M2 Air芯片内存优化配置
        "per_device_eval_batch_size": 1,  # 验证批次大小：M2 Air用1，最小内存占用
        "per_device_train_batch_size": 1,  # 训练批次大小：M2 Air用1，最小内存占用
        "remove_unused_columns": True,  # 移除未使用的列节省内存
        # 保存和评估频率（调试用）
        "save_steps": 1,  # 每步保存：便于调试和查看进度
        "save_total_limit": 3,  # 保留3个检查点
        "eval_strategy": "steps",  # 按步骤进行评估
        "eval_steps": 1,  # 评估频率：匹配保存频率
        "load_best_model_at_end": True,  # 训练结束时加载最佳模型
        "metric_for_best_model": "eval_loss",  # 以验证损失作为最佳模型指标
        # 内存和性能优化
        "gradient_checkpointing": True,  # 启用梯度检查点：节省内存但增加计算
        "gradient_checkpointing_kwargs": {"use_reentrant": False},  # 梯度检查点配置
        "gradient_accumulation_steps": 8,  # 梯度累积：8步累积后更新，模拟批次大小8，M2 Air友好
        "warmup_ratio": 0.1,  # 学习率预热：前10%步骤逐渐增加学习率
        # 其他配置
        "seed": 42,  # 随机种子：确保结果可重现
    }

    # LoRA配置 - 参数高效微调配置
    peft_config = {
        # LoRA核心参数：低秩矩阵维度 - M2 Air优化
        "r": 4,  # LoRA秩：M2 Air用4，极小内存占用（约40万参数）
        # 缩放参数：控制LoRA适配器的强度
        "lora_alpha": 16,  # 缩放因子：保持r的2倍，适应较小的r值
        # 正则化参数：防止过拟合
        "lora_dropout": 0.05,  # LoRA层dropout率：小数据集用较小值防止过拟合
        # 偏置参数：是否训练偏置项
        "bias": "none",  # 偏置设置："none"=不训练，"all"=全部训练
        # 任务类型：模型架构类型
        "task_type": "CAUSAL_LM",  # 任务类型：因果语言模型（自回归生成）
        # 目标模块：要应用LoRA的模型层
        "target_modules": [  # Phi模型的关键注意力层
            "q_proj",  # 查询投影层
            "k_proj",  # 键投影层
            "v_proj",  # 值投影层
            "o_proj",  # 输出投影层
        ],
        # 额外保存模块：除了LoRA适配器外还要保存的层
        "modules_to_save": None,  # 额外保存模块：None=只保存LoRA参数
    }

    # 创建训练参数
    training_args = TrainingArguments(**training_config)
    peft_args = LoraConfig(**peft_config)

    # 创建训练器
    trainer = SFTTrainer(
        model=model,
        args=training_args,
        peft_config=peft_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
    )

    # 开始训练
    logger.info("开始训练...")
    logger.info(f"训练数据大小: {len(train_dataset)}")
    logger.info(f"验证数据大小: {len(eval_dataset) if eval_dataset else 0}")
    logger.info(
        f"总训练步数: {trainer.args.max_steps if trainer.args.max_steps > 0 else 'auto'}"
    )

    # 手动执行训练步骤，显示详细进度
    logger.info("开始执行训练...")

    # 添加更详细的进度跟踪
    import time

    start_time = time.time()

    # 获取模型设备信息
    try:
        device = next(model.parameters()).device
        logger.info(f"模型设备: {device}")
    except:
        logger.info("无法获取模型设备信息")

    # 计算参数数量
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"模型总参数: {total_params:,}")
    logger.info(f"可训练参数: {trainable_params:,}")

    logger.info("开始调用trainer.train()...")
    train_result = trainer.train()

    end_time = time.time()
    logger.info(f"训练总耗时: {end_time - start_time:.2f}秒")

    # 显示训练结果
    logger.info("训练完成！")
    logger.info(f"最终训练损失: {train_result.training_loss:.6f}")
    logger.info(f"总训练步数: {train_result.global_step}")

    # 记录训练指标
    metrics = train_result.metrics
    trainer.log_metrics("train", metrics)
    trainer.save_metrics("train", metrics)

    # 保存模型
    trainer.save_model()
    logger.info(f"模型已保存到 {training_args.output_dir}")

    # 如果有验证集，进行评估
    if eval_dataset is not None:
        logger.info("开始评估...")
        tokenizer.padding_side = "left"  # 评估时用左侧填充
        eval_metrics = trainer.evaluate()
        eval_metrics["eval_samples"] = len(eval_dataset)
        trainer.log_metrics("eval", eval_metrics)
        trainer.save_metrics("eval", eval_metrics)


if __name__ == "__main__":
    main()
