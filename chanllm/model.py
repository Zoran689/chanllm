"""
ChanLLM 模型加载与推理
"""

import os
import torch
from typing import Optional, List, Dict, Generator, Union
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

from .config import Config


class ChanLLM:
    """ChanLLM 缠论大模型"""
    
    def __init__(
        self,
        lora_path: Optional[str] = None,
        base_model: Optional[str] = None,
        device: str = "auto",
        torch_dtype: str = "float16",
        cache_dir: Optional[str] = None,
    ):
        """
        初始化 ChanLLM
        
        Args:
            lora_path: LoRA 适配器路径（本地路径或HuggingFace模型ID）
            base_model: 基础模型路径（默认 Qwen3.5-0.8B）
            device: 设备 (auto, mps, cuda, cpu)
            torch_dtype: 数据类型 (float16, float32, bfloat16)
            cache_dir: 模型缓存目录
        """
        self.config = Config(
            lora_model=lora_path or "zoransun/chanllm-lora",
            base_model=base_model or "Qwen/Qwen3.5-0.8B",
            device=device,
            torch_dtype=torch_dtype,
            cache_dir=cache_dir,
        )
        
        self.model = None
        self.tokenizer = None
        self.device = None
        
        self._load_model()
    
    def _load_model(self):
        """加载模型"""
        # 确定设备
        if self.config.device == "auto":
            if torch.backends.mps.is_available():
                self.device = torch.device('mps')
            elif torch.cuda.is_available():
                self.device = torch.device('cuda')
            else:
                self.device = torch.device('cpu')
        else:
            self.device = torch.device(self.config.device)
        
        # 确定数据类型
        dtype_map = {
            "float16": torch.float16,
            "float32": torch.float32,
            "bfloat16": torch.bfloat16,
        }
        torch_dtype = dtype_map.get(self.config.torch_dtype, torch.float16)
        
        # 加载 tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.config.base_model,
            trust_remote_code=True,
            cache_dir=self.config.cache_dir,
        )
        
        # 加载基础模型
        self.model = AutoModelForCausalLM.from_pretrained(
            self.config.base_model,
            torch_dtype=torch_dtype,
            device_map="auto",
            trust_remote_code=True,
            cache_dir=self.config.cache_dir,
        )
        
        # 加载 LoRA 适配器
        self.model = PeftModel.from_pretrained(
            self.model,
            self.config.lora_model,
            torch_dtype=torch_dtype,
            cache_dir=self.config.cache_dir,
        )
        
        self.model.eval()
    
    def generate(
        self,
        prompt: str,
        history: Optional[List[Dict[str, str]]] = None,
        max_new_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
    ) -> str:
        """
        生成回复
        
        Args:
            prompt: 用户输入
            history: 对话历史 [{"user": "...", "assistant": "..."}]
            max_new_tokens: 最大生成长度
            temperature: 温度参数
            top_p: Top-p 采样参数
            
        Returns:
            生成的回复文本
        """
        # 构建消息
        messages = []
        if history:
            for h in history:
                messages.append({"role": "user", "content": h.get("user", "")})
                messages.append({"role": "assistant", "content": h.get("assistant", "")})
        messages.append({"role": "user", "content": prompt})
        
        # 应用聊天模板
        text = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = self.tokenizer(text, return_tensors="pt").to(self.model.device)
        
        # 生成参数
        gen_kwargs = {
            "max_new_tokens": max_new_tokens or self.config.max_new_tokens,
            "temperature": temperature or self.config.temperature,
            "top_p": top_p or self.config.top_p,
            "do_sample": True,
            "pad_token_id": self.tokenizer.eos_token_id,
            "eos_token_id": self.tokenizer.eos_token_id,
        }
        
        # 生成
        with torch.no_grad():
            outputs = self.model.generate(**inputs, **gen_kwargs)
        
        # 解码
        response = self.tokenizer.decode(
            outputs[0][inputs['input_ids'].shape[1]:],
            skip_special_tokens=True
        )
        return response
    
    def stream_generate(
        self,
        prompt: str,
        history: Optional[List[Dict[str, str]]] = None,
        max_new_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
    ) -> Generator[str, None, None]:
        """
        流式生成回复
        
        Yields:
            生成的文本片段
        """
        from transformers import TextIteratorStreamer
        import threading
        
        # 构建消息
        messages = []
        if history:
            for h in history:
                messages.append({"role": "user", "content": h.get("user", "")})
                messages.append({"role": "assistant", "content": h.get("assistant", "")})
        messages.append({"role": "user", "content": prompt})
        
        # 应用聊天模板
        text = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = self.tokenizer(text, return_tensors="pt").to(self.model.device)
        
        # 流式生成器
        streamer = TextIteratorStreamer(
            self.tokenizer, skip_prompt=True, skip_special_tokens=True
        )
        
        gen_kwargs = {
            **inputs,
            "max_new_tokens": max_new_tokens or self.config.max_new_tokens,
            "temperature": temperature or self.config.temperature,
            "top_p": top_p or self.config.top_p,
            "do_sample": True,
            "pad_token_id": self.tokenizer.eos_token_id,
            "eos_token_id": self.tokenizer.eos_token_id,
            "streamer": streamer,
        }
        
        # 后台线程生成
        thread = threading.Thread(target=self.model.generate, kwargs=gen_kwargs)
        thread.start()
        
        # 流式输出
        for text in streamer:
            yield text
        
        thread.join()
    
    def chat(
        self,
        prompt: str,
        history: Optional[List[Dict[str, str]]] = None,
        stream: bool = False,
        **kwargs,
    ) -> Union[str, Generator[str, None, None]]:
        """
        对话接口
        
        Args:
            prompt: 用户输入
            history: 对话历史
            stream: 是否流式输出
            **kwargs: 其他生成参数
            
        Returns:
            回复文本或生成器
        """
        if stream:
            return self.stream_generate(prompt, history, **kwargs)
        else:
            return self.generate(prompt, history, **kwargs)
    
    def __call__(self, prompt: str, **kwargs) -> str:
        """简化调用"""
        return self.generate(prompt, **kwargs)


def load_model(
    lora_path: Optional[str] = None,
    base_model: Optional[str] = None,
    **kwargs,
) -> ChanLLM:
    """
    加载 ChanLLM 模型
    
    Args:
        lora_path: LoRA 适配器路径
        base_model: 基础模型路径
        **kwargs: 其他参数
        
    Returns:
        ChanLLM 实例
    """
    return ChanLLM(lora_path=lora_path, base_model=base_model, **kwargs)
