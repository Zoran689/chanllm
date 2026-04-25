"""
ChanLLM 命令行接口
"""

import argparse
import sys
from .model import ChanLLM


def main():
    """命令行入口"""
    parser = argparse.ArgumentParser(
        description="ChanLLM 缠论大模型",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  # 交互式对话
  chanllm chat
  
  # 单次问答
  chanllm ask "什么是缠论的中枢？"
  
  # 启动WebUI
  chanllm webui --port 7862
        """
    )
    
    subparsers = parser.add_subparsers(dest="command", help="命令")
    
    # chat 命令
    chat_parser = subparsers.add_parser("chat", help="交互式对话")
    chat_parser.add_argument("--lora", help="LoRA适配器路径")
    chat_parser.add_argument("--base", help="基础模型路径")
    chat_parser.add_argument("--max-tokens", type=int, default=1024, help="最大生成长度")
    
    # ask 命令
    ask_parser = subparsers.add_parser("ask", help="单次问答")
    ask_parser.add_argument("question", help="问题")
    ask_parser.add_argument("--lora", help="LoRA适配器路径")
    ask_parser.add_argument("--base", help="基础模型路径")
    ask_parser.add_argument("--max-tokens", type=int, default=512, help="最大生成长度")
    
    # webui 命令
    webui_parser = subparsers.add_parser("webui", help="启动WebUI")
    webui_parser.add_argument("--port", type=int, default=7862, help="端口号")
    webui_parser.add_argument("--host", default="127.0.0.1", help="主机地址")
    webui_parser.add_argument("--share", action="store_true", help="创建公网链接")
    webui_parser.add_argument("--lora", help="LoRA适配器路径")
    webui_parser.add_argument("--base", help="基础模型路径")
    
    args = parser.parse_args()
    
    if args.command is None:
        parser.print_help()
        sys.exit(1)
    
    if args.command == "chat":
        interactive_chat(args)
    elif args.command == "ask":
        single_ask(args)
    elif args.command == "webui":
        start_webui(args)


def interactive_chat(args):
    """交互式对话"""
    print("加载模型中...")
    llm = ChanLLM(lora_path=args.lora, base_model=args.base)
    
    print("\n" + "=" * 60)
    print("ChanLLM 缠论大模型 - 交互式对话")
    print("输入 'quit' 退出, 'clear' 清空历史")
    print("=" * 60 + "\n")
    
    history = []
    
    while True:
        try:
            user_input = input("用户: ").strip()
            
            if not user_input:
                continue
            
            if user_input.lower() == 'quit':
                print("\n再见！")
                break
            
            if user_input.lower() == 'clear':
                history = []
                print("已清空对话历史\n")
                continue
            
            print("助手: ", end="", flush=True)
            response = llm.generate(user_input, history, max_new_tokens=args.max_tokens)
            print(response)
            print()
            
            history.append({"user": user_input, "assistant": response})
            
        except KeyboardInterrupt:
            print("\n\n再见！")
            break


def single_ask(args):
    """单次问答"""
    llm = ChanLLM(lora_path=args.lora, base_model=args.base)
    response = llm.generate(args.question, max_new_tokens=args.max_tokens)
    print(response)


def start_webui(args):
    """启动WebUI"""
    from .webui import launch_webui
    
    launch_webui(
        lora_path=args.lora,
        base_model=args.base,
        server_name=args.host,
        server_port=args.port,
        share=args.share,
    )


if __name__ == "__main__":
    main()
