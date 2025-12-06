# M2芯片Mac上Phi-3.5模型微调完整指南

## 环境配置

### 1. 创建虚拟环境
```bash
python -m venv .venv
source .venv/bin/activate  # Mac/Linux
```

### 2. 安装依赖
```bash
pip install torch transformers datasets peft trl accelerate
pip install black ruff isort  # 代码格式化工具
```

### 3. VS Code配置 (.vscode/settings.json)
```json
{
    "[python]": {
        "editor.defaultFormatter": "ms-python.black-formatter",
        "editor.formatOnSave": true,
        "editor.codeActionsOnSave": {
            "source.organizeImports": "explicit",
            "source.fixAll": "explicit"
        },
        "editor.insertSpaces": true,
        "editor.tabSize": 4
    },
    "editor.formatOnSave": true,
    "editor.insertSpaces": true,
    "editor.tabSize": 4,
    "python.defaultInterpreterPath": "${workspaceFolder}/.venv/bin/python",
    "python.terminal.activateEnvironment": true
}
```

## 核心问题解决方案

### 问题1：VS Code保存时不会自动格式化

**原因**：
- 未安装Black Formatter扩展
- Python解释器选择错误
- 配置参数错误

**解决方案**：
1. 安装VS Code扩展：
   - Python (ms-python.python)
   - Black Formatter (ms-python.black-formatter)

2. 选择正确的Python解释器：
   - `Cmd+Shift+P` → "Python: Select Interpreter"
   - 选择 `{workspaceFolder}/.venv/bin/python`

3. 配置settings.json（如上所示）

### 问题2：Tab和空格混合导致的语法错误

**原因**：
- Python不允许tab和空格混合缩进
- Black不会处理有语法错误的文件

**解决方案**：
1. **预防**：在settings.json中配置
   ```json
   "editor.insertSpaces": true,
   "editor.tabSize": 4
   ```

2. **修复现有问题**：
   - `Cmd+Shift+P` → "Convert Indentation to Spaces"
   - 手动删除tab字符，重新用空格缩进

### 问题3：M2芯片设备配置问题

**错误信息**：
```
RuntimeError: Function MmBackward0 returned an invalid gradient at index 1 - expected device meta but got cpu
```

**原因**：
- `device_map="auto"`在M系列芯片上会导致参数分散到不同设备
- 梯度计算时设备不匹配

**解决方案**：
```python
# 模型配置参数 - M2芯片优化配置
model_kwargs = {
    "use_cache": False,
    "trust_remote_code": True,
    "torch_dtype": torch.float16,
    # 关键：不使用device_map="auto"
    # device_map="auto",  # 注释掉
    "attn_implementation": "eager",
}

# 手动移动到MPS设备
model = AutoModelForCausalLM.from_pretrained(model_name, **model_kwargs)
if torch.backends.mps.is_available():
    model = model.to("mps")
```

### 问题4：参数名称错误

**常见错误及修复**：

1. **`evaluation_strategy` → `eval_strategy`**
```python
training_config = {
    "eval_strategy": "steps",  # 新版本使用这个参数名
}
```

2. **`torch_dtype` is deprecated**
```python
model_kwargs = {
    "torch_dtype": torch.float16,  # 使用这个
    # 而不是 dtype
}
```

3. **`max_seq_length` → `max_seq_len`**
```python
trainer = SFTTrainer(
    # max_seq_length=2048,  # 错误
    max_seq_len=2048,        # 正确
)
```

4. **移除不支持的参数**
```python
trainer = SFTTrainer(
    model=model,
    args=training_args,
    peft_config=peft_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    # 移除这些参数：
    # dataset_text_field="text",
    # tokenizer=tokenizer,
    # max_seq_len=2048,
)
```

### 问题5：bfloat16不支持错误

**错误信息**：
```
ValueError: Your setup doesn't support bf16/gpu.
```

**原因**：M2芯片不支持bfloat16

**解决方案**：
```python
# 方案1：使用fp16（如果支持）
training_config = {
    "fp16": True,
}

# 方案2：完全不使用混合精度（推荐M2）
training_config = {
    # 注释掉fp16
    # "fp16": True,
}
```

## 完整的训练代码模板

### 1. 数据准备
```python
def prepare_dataset(tokenizer, data_path=None, use_sample_data=True):
    """准备训练数据集"""

    # 步骤1：创建或加载数据
    if use_sample_data:
        data_path = create_sample_alpaca_data()

    # 步骤2：转换Alpaca格式为消息格式
    messages_data = convert_alpaca_to_messages_format(data_path)

    # 步骤3：创建Hugging Face数据集
    raw_dataset = Dataset.from_list(messages_data)

    # 步骤4：应用聊天模板
    def apply_chat_template(example):
        example["text"] = tokenizer.apply_chat_template(
            example["messages"],
            tokenize=False,
            add_generation_prompt=False,
        )
        return example

    processed_dataset = raw_dataset.map(apply_chat_template, remove_columns=raw_dataset.column_names)

    # 步骤5：分割数据集
    if len(processed_dataset) > 1:
        dataset = processed_dataset.train_test_split(test_size=0.2, seed=42)
        return dataset["train"], dataset["test"]
    else:
        return processed_dataset, None
```

### 2. 模型加载（M2优化）
```python
def load_model_and_tokenizer(model_name="microsoft/Phi-3.5-mini-instruct"):
    """M2芯片优化的模型加载"""

    model_kwargs = {
        "use_cache": False,
        "trust_remote_code": True,
        "torch_dtype": torch.float16,
        "attn_implementation": "eager",
    }

    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.unk_token
    tokenizer.padding_side = "right"

    model = AutoModelForCausalLM.from_pretrained(model_name, **model_kwargs)

    # M2芯片：移动到MPS设备
    if torch.backends.mps.is_available():
        model = model.to("mps")
        logger.info("模型已移动到MPS设备（M2芯片）")
    else:
        logger.info("使用CPU训练")

    return model, tokenizer
```

### 3. 训练配置
```python
def main():
    setup_logging()

    # 加载模型和数据
    model, tokenizer = load_model_and_tokenizer()
    data_path = create_sample_alpaca_data()
    train_dataset, eval_dataset = prepare_dataset(tokenizer, data_path)

    # 训练配置 - M2芯片优化
    training_config = {
        "do_eval": True,
        "learning_rate": 5.0e-06,
        "log_level": "info",
        "logging_steps": 1,
        "logging_strategy": "steps",
        "lr_scheduler_type": "cosine",
        "num_train_epochs": 3,
        "max_steps": -1,
        "output_dir": "./phi_checkpoint",
        "overwrite_output_dir": True,
        "per_device_eval_batch_size": 2,
        "per_device_train_batch_size": 2,
        "remove_unused_columns": True,
        "save_steps": 50,
        "save_total_limit": 1,
        "seed": 42,
        "gradient_checkpointing": True,
        "gradient_checkpointing_kwargs": {"use_reentrant": False},
        "gradient_accumulation_steps": 4,
        "warmup_ratio": 0.1,
        "eval_strategy": "steps",
        "eval_steps": 50,
        "load_best_model_at_end": True,
        "metric_for_best_model": "eval_loss",
    }

    # LoRA配置
    peft_config = {
        "r": 16,
        "lora_alpha": 32,
        "lora_dropout": 0.05,
        "bias": "none",
        "task_type": "CAUSAL_LM",
        "target_modules": ["q_proj", "k_proj", "v_proj", "o_proj"],
        "modules_to_save": None,
    }

    # 创建训练器
    training_args = TrainingArguments(**training_config)
    peft_args = LoraConfig(**peft_config)

    trainer = SFTTrainer(
        model=model,
        args=training_args,
        peft_config=peft_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
    )

    # 开始训练
    logger.info("开始训练...")
    train_result = trainer.train()

    # 保存模型
    trainer.save_model()
    logger.info(f"模型已保存到 {training_args.output_dir}")

    # 评估
    if eval_dataset is not None:
        tokenizer.padding_side = 'left'
        eval_metrics = trainer.evaluate()
        trainer.log_metrics("eval", eval_metrics)
```

## 常见警告信息处理

### 1. Flash Attention警告
```
`flash-attention` package not found
```
**解决方案**：在模型配置中使用`attn_implementation="eager"`

### 2. Pin Memory警告
```
'pin_memory' argument is set as true but no accelerator is found
```
**解决方案**：这是正常警告，不影响训练，可以忽略

### 3. dtype警告
```
`torch_dtype` is deprecated! Use `dtype` instead!
```
**解决方案**：目前仍使用`torch_dtype`参数，新版本可能需要调整

## 性能优化建议

### 1. M2芯片特有优化
- 使用MPS设备而非CPU
- 避免不必要的设备切换
- 使用适当的批次大小（通常2-4）

### 2. 内存优化
- 启用梯度检查点：`gradient_checkpointing=True`
- 使用梯度累积：`gradient_accumulation_steps=4`
- 选择合适的数据类型：`torch.float16`

### 3. 训练稳定性
- 使用较小的学习率：`5e-06`
- 启用预热：`warmup_ratio=0.1`
- 保存检查点：`save_steps=50`

## 故障排查清单

- [ ] VS Code扩展已安装
- [ ] Python解释器正确选择
- [ ] settings.json配置正确
- [ ] 依赖包版本兼容
- [ ] M2芯片特殊配置已应用
- [ ] 数据格式正确
- [ ] 模型加载成功
- [ ] 训练参数设置合理

## 扩展阅读

- [Hugging Face TRL文档](https://huggingface.co/docs/trl)
- [PEFT参数高效微调](https://huggingface.co/docs/peft)
- [Phi-3.5模型文档](https://huggingface.co/microsoft/Phi-3.5-mini-instruct)