# ChanLLM 缠论大模型

基于 **Qwen3.5-0.8B + LoRA** 微调的缠论专业知识模型。

## 特性

- 🎯 **专业缠论知识**：训练于173个缠论文档，2,257条SFT数据
- 🚀 **轻量高效**：0.8B基础模型 + 10.8M LoRA参数
- 💬 **流式输出**：支持逐字显示回答
- 🖥️ **多种接口**：CLI命令行、WebUI、Python API

## 安装

### 从 GitHub 安装

```bash
pip install git+https://github.com/Zoran689/chanllm.git
```

### 从源码安装

```bash
git clone https://github.com/Zoran689/chanllm.git
cd chanllm
pip install -e .
```

## 快速开始

### 1. 命令行对话

```bash
# 交互式对话
chanllm chat

# 单次问答
chanllm ask "什么是缠论的中枢？"
```

### 2. 启动 WebUI

```bash
# 默认端口 7862
chanllm webui

# 自定义端口
chanllm webui --port 8080

# 创建公网链接
chanllm webui --share
```

### 3. Python API

```python
from chanllm import ChanLLM

# 加载模型
llm = ChanLLM()

# 单次问答
response = llm("什么是缠论的中枢？")
print(response)

# 流式输出
for chunk in llm.stream_generate("如何判断背驰？"):
    print(chunk, end="", flush=True)

# 多轮对话
history = []
while True:
    user_input = input("用户: ")
    response = llm.generate(user_input, history)
    print(f"助手: {response}")
    history.append({"user": user_input, "assistant": response})
```

## 模型信息

| 项目 | 值 |
|------|-----|
| 基础模型 | Qwen3.5-0.8B |
| LoRA参数 | 10.8M (r=16, alpha=32) |
| 训练数据 | 173个缠论文档，2,257条SFT |
| 训练步数 | 426步 |
| 训练时长 | 12小时28分钟 |
| 最终Loss | 0.27 |

## 依赖

- Python >= 3.9
- PyTorch >= 2.0
- Transformers >= 4.40
- PEFT >= 0.10
- Gradio >= 5.0 (WebUI)

## 许可证

MIT License

## 致谢

- [Qwen](https://github.com/QwenLM/Qwen) - 基础模型
- [PEFT](https://github.com/huggingface/peft) - LoRA微调框架
