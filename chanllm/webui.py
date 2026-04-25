"""
ChanLLM WebUI
基于 Gradio 构建
"""

import gradio as gr
from typing import Optional
from .model import ChanLLM


def launch_webui(
    lora_path: Optional[str] = None,
    base_model: Optional[str] = None,
    server_name: str = "127.0.0.1",
    server_port: int = 7862,
    share: bool = False,
):
    """
    启动 WebUI
    
    Args:
        lora_path: LoRA 适配器路径
        base_model: 基础模型路径
        server_name: 服务器地址
        server_port: 端口号
        share: 是否创建公网链接
    """
    print("加载模型中...")
    llm = ChanLLM(lora_path=lora_path, base_model=base_model)
    print("模型加载完成！")
    
    def chat(message, history, temperature, top_p, max_tokens):
        """聊天函数 - 流式输出"""
        if not message.strip():
            yield history, ""
            return
        
        # 转换历史格式
        chat_history = []
        if history:
            i = 0
            while i < len(history):
                if history[i].get("role") == "user":
                    user_msg = history[i].get("content", "")
                    assistant_msg = history[i+1].get("content", "") if i+1 < len(history) and history[i+1].get("role") == "assistant" else ""
                    chat_history.append({"user": user_msg, "assistant": assistant_msg})
                    i += 2
                else:
                    i += 1
        
        # 添加用户消息
        new_history = history.copy() if history else []
        new_history.append({"role": "user", "content": message})
        
        # 流式生成
        full_response = ""
        for chunk in llm.stream_generate(
            message,
            history=chat_history if chat_history else None,
            max_new_tokens=int(max_tokens),
            temperature=temperature,
            top_p=top_p
        ):
            full_response += chunk
            display_history = new_history.copy()
            display_history.append({"role": "assistant", "content": full_response})
            yield display_history, ""
        
        final_history = new_history.copy()
        final_history.append({"role": "assistant", "content": full_response})
        yield final_history, ""
    
    # 创建界面
    with gr.Blocks(title="ChanLLM 缠论大模型") as demo:
        gr.Markdown("""
        # 🎯 ChanLLM 缠论大模型
        
        基于 **Qwen3.5-0.8B + LoRA** 微调的缠论专业知识模型
        
        - **参数量**: 0.8B 基础 + 10.8M LoRA
        - **训练数据**: 173个缠论文档，2,257条SFT数据
        """)
        
        with gr.Row():
            with gr.Column(scale=3):
                chatbot = gr.Chatbot(label="对话", height=500)
                
                with gr.Row():
                    msg = gr.Textbox(
                        label="输入问题",
                        placeholder="例如：什么是缠论的中枢？",
                        lines=2,
                        scale=4
                    )
                    submit_btn = gr.Button("发送", variant="primary", scale=1)
                
                with gr.Row():
                    clear_btn = gr.Button("清空对话", variant="secondary")
            
            with gr.Column(scale=1):
                with gr.Group():
                    gr.Markdown("### 生成参数")
                    temperature = gr.Slider(0.1, 2.0, value=0.8, label="Temperature")
                    top_p = gr.Slider(0.1, 1.0, value=0.9, label="Top P")
                    max_tokens = gr.Slider(256, 2048, value=1024, step=64, label="最大生成长度")
        
        # 示例问题
        examples = [
            "什么是缠论的中枢？",
            "缠论中笔和线段有什么区别？",
            "如何判断缠论中的背驰？",
            "缠论的买卖点有哪些类型？",
        ]
        
        gr.Examples(examples=examples, inputs=msg, label="示例问题")
        
        # 事件绑定
        submit_btn.click(chat, [msg, chatbot, temperature, top_p, max_tokens], [chatbot, msg])
        msg.submit(chat, [msg, chatbot, temperature, top_p, max_tokens], [chatbot, msg])
        clear_btn.click(lambda: ([], ""), outputs=[chatbot, msg])
        
        gr.Markdown("""
        ---
        **GitHub**: https://github.com/zoransun/chanllm
        """)
    
    # 启动
    demo.launch(
        server_name=server_name,
        server_port=server_port,
        share=share,
        show_error=True,
    )