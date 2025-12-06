# 微调后模型使用指南

## 模型位置

你的微调模型位于以下位置：

### 主位置：`./phi_checkpoint/`

这个目录包含了：
- `adapter_model.safetensors` - LoRA适配器权重文件
- `adapter_config.json` - LoRA配置信息
- `tokenizer.json` 和 `tokenizer_config.json` - 分词器文件
- `special_tokens_map.json` - 特殊token映射
- `chat_template.jinja` - 聊天模板

### 检查点位置：
- `./phi_checkpoint/checkpoint-1/`
- `./phi_checkpoint/checkpoint-2/`
- `./phi_checkpoint/checkpoint-3/`

## 模型信息

- **基础模型**: `microsoft/Phi-3.5-mini-instruct`
- **微调技术**: LoRA (Low-Rank Adaptation)
- **任务类型**: Causal Language Modeling
- **LoRA参数**:
  - r=4 (低秩维度)
  - alpha=16 (缩放因子)
  - dropout=0.05
  - 目标模块: ['q_proj', 'k_proj', 'v_proj', 'o_proj']


## 使用方法

### 1. 加载模型

```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

def load_finetuned_model():
    # 加载tokenizer
    tokenizer = AutoTokenizer.from_pretrained("microsoft/Phi-3.5-mini-instruct", trust_remote_code=True)

    # 加载基础模型
    model = AutoModelForCausalLM.from_pretrained(
        "microsoft/Phi-3.5-mini-instruct",
        torch_dtype=torch.float16,
        trust_remote_code=True,
        device_map="auto"
    )

    # 加载LoRA适配器
    model = PeftModel.from_pretrained(model, "./phi_checkpoint")

    return model, tokenizer
```

### 2. 生成回复

```python
def generate_response(model, tokenizer, prompt):
    # 构建对话格式
    messages = [{"role": "user", "content": prompt}]

    # 应用聊天模板
    text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )

    # 编码并生成
    inputs = tokenizer(text, return_tensors="pt")
    outputs = model.generate(
        **inputs,
        max_new_tokens=512,
        temperature=0.7,
        do_sample=True
    )

    # 解码回复
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return response.replace(text, "").strip()
```

### 3. 使用pipeline方式

```python
from transformers import pipeline

# 创建pipeline
generator = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    device="mps"  # 或 "cuda" / "cpu"
)

# 生成回复
messages = [{"role": "user", "content": "你的问题"}]
output = generator(messages, max_new_tokens=128)[0]
print(output["generated_text"])
```

## 重要说明

1. **参数高效微调**: 你使用的是LoRA技术，只保存了适配器权重，不是完整模型
2. **加载顺序**: 必须先加载基础模型，再加载LoRA适配器
3. **设备选择**: 模型会自动选择可用设备(MPS/CUDA/CPU)
4. **内存占用**: LoRA适配器相对较小，适合在有限内存环境中使用

## 完整使用示例

参考项目中的 `use_model.py` 文件，它包含了：
- 模型加载
- 交互式聊天
- 示例测试
- 错误处理

运行方式：
```bash
python use_model.py
```