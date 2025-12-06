## 环境配置 (使用 UV)

如果你使用 `uv` 作为包管理器，以下是环境配置和使用方法：

### 1. 简单使用方式（推荐）

```bash
# 直接运行脚本，uv会自动创建和管理虚拟环境
uv run python use_model.py
```

**说明**：
- `uv run` 会自动检测或创建虚拟环境
- 如果项目中有 `pyproject.toml` 或 `requirements.txt`，会自动安装依赖
- 这是使用 `uv` 最简单的方式

### 2. 手动管理依赖

```bash
# 添加依赖包到项目
uv add torch torchvision torchaudio
uv add transformers
uv add peft
uv add datasets
uv add accelerate
uv add trl

# 查看项目依赖
uv tree

# 运行脚本
uv run python use_model.py

# 更新所有包
uv sync
```

### 3. 创建独立虚拟环境

```bash
# 创建新的虚拟环境
uv venv

# 激活环境
source .venv/bin/activate

# 或者使用uv激活
uv shell

# 安装依赖
uv add torch transformers peft datasets accelerate trl

# 运行脚本
python use_model.py

```

### 4. 常用uv命令

```bash
# 查看已安装的包
uv pip list

# 更新包
uv add --upgrade transformers

# 移除包
uv remove peft

# 导出依赖
uv pip freeze > requirements.txt

# 从requirements.txt安装
uv pip install -r requirements.txt

# 清理缓存
uv cache clean
```
