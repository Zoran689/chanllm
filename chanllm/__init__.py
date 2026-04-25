"""
ChanLLM - 缠论大模型
基于 Qwen3.5-0.8B + LoRA 微调的缠论专业知识模型
"""

__version__ = "1.0.0"
__author__ = "zoransun"

from .model import ChanLLM, load_model
from .config import Config

__all__ = ["ChanLLM", "load_model", "Config"]
