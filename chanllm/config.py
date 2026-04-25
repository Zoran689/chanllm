"""
ChanLLM 配置
"""

import os
from dataclasses import dataclass
from typing import Optional


@dataclass
class Config:
    """ChanLLM 配置类"""
    
    # 基础模型
    base_model: str = "Qwen/Qwen3.5-0.8B"
    
    # LoRA 适配器路径（本地或HuggingFace）
    lora_model: str = "zoransun/chanllm-lora"  # HuggingFace Hub
    
    # 本地缓存路径
    cache_dir: Optional[str] = None
    
    # 生成参数
    max_new_tokens: int = 1024
    temperature: float = 0.8
    top_p: float = 0.9
    
    # 设备
    device: str = "auto"  # auto, mps, cuda, cpu
    
    # 数据类型
    torch_dtype: str = "float16"
    
    def __post_init__(self):
        """初始化后处理"""
        if self.cache_dir is None:
            self.cache_dir = os.path.expanduser("~/.cache/chanllm")
