首先toml文件：
```
[build-system]
requires = ["setuptools>=61.0", "wheel"]
build-backend = "setuptools.build_meta"

[project]
name = "qc-train-phi"
version = "0.1.0"
description = "Phi fine-tune on Mac M2/M3/M4 with stable MPS"
readme = "README.md"
requires-python = ">=3.10"
dependencies = [
    "datasets>=2.14.0",
    "transformers>=4.46.0",
    "peft>=0.18.0",
    "trl>=0.25.1",
    "black>=24.0.0",
    "ruff>=0.7.0",
    "isort>=5.0.0",
]

# Mac Apple Silicon 强制锁最稳 MPS 版
[project.optional-dependencies]
mps = [
    "torch==2.4.1; sys_platform == 'darwin' and platform_machine == 'arm64'",
    "torchvision==0.19.1; sys_platform == 'darwin' and platform_machine == 'arm64'",
    "torchaudio==2.4.1; sys_platform == 'darwin' and platform_machine == 'arm64'",
]
cpu = [
    "torch; sys_platform != 'darwin' or platform_machine != 'arm64'",
    "torchvision; sys_platform != 'darwin' or platform_machine != 'arm64'",
    "torchaudio; sys_platform != 'darwin' or platform_machine != 'arm64'",
]

# 告诉 setuptools 忽略数据文件夹
[tool.setuptools]
packages = []  # 空列表，不包含任何包
include-package-data = false

```

然后运行
```
 conda deactivate; conda deactivate && rm -rf .venv uv.lock && python3 -m venv .venv && source .venv/bin/activate && pip install --upgrade pip && pip install -e . --extra-index-url https://download.pytorch.org/whl/cpu --no-cache-dir

```
### 这一行实际上用到了pyproject.toml
  pip install -e . --extra-index-url https://download.pytorch.org/whl/cpu --no-cache-dir


### MPS 问题背景

  MPS (Metal Performance Shaders) 是苹果 M 系列芯片上的 GPU 加速框架。在使用 PyTorch 时，经常会遇到 MPS
  相关的问题，特别是在 Mac M1/M2/M3 芯片上。

  解决方案分析

  1. 关键：PyTorch 版本锁定

  mps = [
      "torch==2.4.1; sys_platform == 'darwin' and platform_machine == 'arm64'",
      "torchvision==0.19.1; sys_platform == 'darwin' and platform_machine == 'arm64'",
      "torchaudio==2.4.1; sys_platform == 'darwin' and platform_machine == 'arm64'",
  ]

  为什么这样有效？
  - 特定版本稳定性：PyTorch 2.4.1 被验证在 Apple Silicon 上运行 MPS 比较稳定
  - 避免版本冲突：新版本可能引入 MPS 兼容性问题
  - 版本协同：torch、torchvision、torchaudio 版本号保持一致，确保组件间兼容

  2. 平台检测机制

  sys_platform == 'darwin' and platform_machine == 'arm64'

  为什么这样有效？
  - darwin: macOS 系统标识
  - arm64: Apple Silicon 架构标识
  - 精确匹配：只在正确的平台安装 MPS 版本，避免在不兼容的系统上安装

  3. 环境重置命令

  conda deactivate; conda deactivate && rm -rf .venv uv.lock && python3 -m venv .venv && source .venv/bin/activate &&
  pip install --upgrade pip && pip install -e . --extra-index-url https://download.pytorch.org/whl/cpu --no-cache-dir

  为什么这样有效？

  a) 完全清理环境

  conda deactivate; conda deactivate  # 退出所有conda环境
  rm -rf .venv uv.lock               # 删除虚拟环境和锁文件

  b) 重新创建纯净环境

  python3 -m venv .venv && source .venv/bin/activate && pip install --upgrade pip

  c) 关键参数解释

  - --extra-index-url https://download.pytorch.org/whl/cpu: 使用 PyTorch 官方源
  - --no-cache-dir: 避免使用缓存的可能有问题的版本
  - -e .: 以开发模式安装当前项目

  核心原理

  这个解决方案有效的原因是：

  1. 版本控制：强制使用已知稳定的 PyTorch 版本
  2. 环境隔离：完全重置 Python 环境，避免版本冲突
  3. 平台适配：根据不同系统选择合适的 PyTorch 构建版本
  4. 缓存清理：确保下载的是正确版本，而不是缓存的旧版本

  为什么不直接 pip install torch

  - 默认安装可能安装最新版本，而最新版本可能有 MPS bug
  - 可能安装的是 CPU 版本而不是 MPS 支持版本
  - 环境中可能存在版本冲突

  这种方法通过严格控制版本和环境，确保在 Mac Apple Silicon 上获得稳定的 MPS 支持。