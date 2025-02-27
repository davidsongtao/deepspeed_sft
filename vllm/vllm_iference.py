"""
Description: DeepSeek模型的命令行交互式聊天界面

-*- Encoding: UTF-8 -*-
@File     ：vllm_inference.py
@Author   ：King Songtao
@Time     ：2025/2/27 下午4:20
@Contact  ：king.songtao@gmail.com
"""
import sys
import time
import traceback

try:
    from vllm.vllm_wrapper import vLLMWrapper
except ImportError:
    from vllm_wrapper import vLLMWrapper


def main():
    print("正在初始化DeepSeek模型，请稍候...")
    start_time = time.time()

    # 模型路径
    model = "/root/autodl-fs/trained_models/deepseek_ri_32b_merged"

    # 关键参数调整
    tensor_parallel_size = 2  # 设置张量并行度
    gpu_memory_utilization = 0.98  # 提高GPU内存使用率
    max_model_len = 32768  # 减小最大序列长度

    try:
        # 初始化模型，添加关键参数
        vllm_model = vLLMWrapper(
            model,
            tensor_parallel_size=tensor_parallel_size,
            gpu_memory_utilization=gpu_memory_utilization,
            max_model_len=max_model_len  # 添加这个参数来解决KV缓存问题
        )

        print(f"模型加载完成，耗时 {time.time() - start_time:.2f} 秒")
        print("\n===== DeepSeek AI 聊天助手 =====")
        print("提示: 输入'quit'或'exit'退出程序, 输入'clear'清除历史记录")
        print("=============================\n")

        # 初始化对话历史
        history = None
        system_prompt = "你是一个有用的AI助手，用中文回答问题。"

        # 交互循环
        while True:
            try:
                # 获取用户输入
                query = input("\n用户: ")

                # 处理特殊命令
                if query.lower() in ["quit", "exit", "q"]:
                    print("再见!")
                    break
                elif query.lower() == "clear":
                    history = None
                    print("已清除对话历史")
                    continue
                elif query.strip() == "":
                    continue

                # 记录开始时间
                gen_start = time.time()

                # 调用模型生成回复
                print("AI思考中...", end="\r")
                response, history = vllm_model.chat(
                    query=query,
                    history=history,
                    system=system_prompt
                )

                # 打印回复和统计信息
                print(f"AI: {response}")
                print(f"[生成时间: {time.time() - gen_start:.2f}秒]")

                # 限制历史长度，防止上下文过长
                if history and len(history) > 10:  # 保留最近10轮对话，防止上下文过长
                    history = history[-10:]

            except KeyboardInterrupt:
                print("\n\n程序被用户中断")
                break
            except Exception as e:
                print(f"\n生成回复时出错: {e}")
                print("请重试或尝试清除历史记录 (输入'clear')")
                traceback.print_exc()

    except Exception as e:
        print(f"模型初始化失败: {e}")
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
