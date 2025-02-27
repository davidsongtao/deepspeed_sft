from transformers import PreTrainedTokenizer, GenerationConfig, StoppingCriteriaList
from typing import Optional, Callable, List, Tuple, Union
import copy
import torch
from transformers import AutoTokenizer
from transformers.generation.logits_process import LogitsProcessorList
from packaging import version

# 定义历史记录类型和token序列类型的别名，以方便后续使用
HistoryType = List[Tuple[str, str]]  # 每个元素是一个元组，包含用户的输入和模型的回复
TokensType = List[int]  # token序列，即文本编码后的整数列表

# 定义消息结束符常量
IMEND = "<|im_end|>"
ENDOFTEXT = "<|endoftext|>"


def get_stop_words_ids(chat_format, tokenizer):
    """
    根据指定的聊天格式(chat_format)和分词器(tokenizer)，获取停止词的ID列表。
    """
    if chat_format == "raw":
        # 如果聊天格式是原始格式，停止词包括"Human:"的token序列和EOD（End of Document）标记的ID。
        stop_words_ids = [tokenizer.encode("Human:")]
        if hasattr(tokenizer, 'eod_id'):
            stop_words_ids.append([tokenizer.eod_id])
    elif chat_format == "chatml":
        # 如果聊天格式是chatml格式，停止词包括IMEND（Assistant消息结束符）的token ID和IMSTART（开始符）的token ID。
        stop_words_ids = [[tokenizer.im_end_id], [tokenizer.im_start_id]]
    else:
        # 如果聊天格式不是已知的格式，抛出NotImplementedError异常。
        raise NotImplementedError(f"Unknown chat format {chat_format!r}")
    return stop_words_ids


def make_deepseek_context(
        tokenizer,
        query: str,
        history: List[Tuple[str, str]] = None,
        system: str = "You are a helpful assistant."
):
    """为DeepSeek模型构建对话上下文"""
    if history is None:
        history = []

    # 构建提示文本
    prompt = ""

    # 添加系统提示
    if system:
        prompt += f"{system}\n\n"

    # 添加历史对话
    for h_query, h_response in history:
        prompt += f"Human: {h_query}\nAssistant: {h_response}\n\n"

    # 添加当前查询
    prompt += f"Human: {query}\nAssistant:"

    # 编码为token
    tokens = tokenizer.encode(prompt)

    return prompt, tokens


def make_context(
        tokenizer: PreTrainedTokenizer,
        query: str,
        history: List[Tuple[str, str]] = None,
        system: str = "",
        max_window_size: int = 6144,
        chat_format: str = "chatml",
):
    """
    准备用于模型生成回复的上下文。
    """
    # 首先尝试使用DeepSeek格式构建上下文
    try:
        return make_deepseek_context(tokenizer, query, history, system)
    except Exception as e:
        print(f"使用DeepSeek格式构建上下文失败: {e}")
        print("尝试使用通用格式...")

    if history is None:
        history = []

    if chat_format == "chatml":
        im_start_tokens = [tokenizer.im_start_id]
        im_end_tokens = [tokenizer.im_end_id]
        im_start, im_end = tokenizer.decode(im_start_tokens, skip_special_tokens=False), tokenizer.decode(im_end_tokens, skip_special_tokens=False)
        nl_tokens = tokenizer.encode("\n")

        def _tokenize_str(role, content):
            """将角色和内容组合成字符串并进行token化。"""
            # 移除 allowed_special 参数，以适应DeepSeek模型
            return f"{role}\n{content}", tokenizer.encode(role) + nl_tokens + tokenizer.encode(content)

        system_text, system_tokens_part = _tokenize_str("system", system)
        system_tokens = im_start_tokens + system_tokens_part + im_end_tokens

        raw_text = ""
        context_tokens = []

        # 反向遍历历史记录，构建上下文
        for turn_query, turn_response in reversed(history):
            query_text, query_tokens_part = _tokenize_str("user", turn_query)
            query_tokens = im_start_tokens + query_tokens_part + im_end_tokens
            response_text, response_tokens_part = _tokenize_str(
                "assistant", turn_response
            )
            response_tokens = im_start_tokens + response_tokens_part + im_end_tokens

            next_context_tokens = nl_tokens + query_tokens + nl_tokens + response_tokens
            prev_chat = (
                f"\n{im_start}{query_text}{im_end}\n{im_start}{response_text}{im_end}"
            )

            current_context_size = (
                    len(system_tokens) + len(next_context_tokens) + len(context_tokens)
            )
            if current_context_size < max_window_size:
                context_tokens = next_context_tokens + context_tokens
                raw_text = prev_chat + raw_text
            else:
                break

        context_tokens = system_tokens + context_tokens
        raw_text = f"{im_start}{system_text}{im_end}" + raw_text
        context_tokens += (
                nl_tokens
                + im_start_tokens
                + _tokenize_str("user", query)[1]
                + im_end_tokens
                + nl_tokens
                + im_start_tokens
                + tokenizer.encode("assistant")
                + nl_tokens
        )
        raw_text += f"\n{im_start}user\n{query}{im_end}\n{im_start}assistant\n"

    elif chat_format == "raw":
        raw_text = query
        context_tokens = tokenizer.encode(raw_text)
    else:
        raise NotImplementedError(f"Unknown chat format {chat_format!r}")

    return raw_text, context_tokens


class vLLMWrapper:
    def __init__(self,
                 model_dir: str,
                 trust_remote_code: bool = True,
                 tensor_parallel_size: int = 1,
                 gpu_memory_utilization: float = 0.98,
                 dtype: str = "bfloat16",
                 max_model_len: int = None,
                 **kwargs):
        """初始化vLLM包装器"""

        if dtype not in ("bfloat16", "float16", "float32"):
            print(f"不支持的数据类型: {dtype}")
            raise ValueError(f"不支持的数据类型: {dtype}")

        # 构建生成配置
        self.generation_config = GenerationConfig.from_pretrained(model_dir, trust_remote_code=trust_remote_code)

        # 添加必要的属性
        if not hasattr(self.generation_config, 'chat_format'):
            print("为生成配置添加默认chat_format: chatml")
            self.generation_config.chat_format = "chatml"

        if not hasattr(self.generation_config, 'max_window_size'):
            print("为生成配置添加默认max_window_size: 6144")
            self.generation_config.max_window_size = 6144

        # 构建分词器
        self.tokenizer = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=True)

        # 确保tokenizer有必要的属性
        if not hasattr(self.tokenizer, 'im_start_id') or not hasattr(self.tokenizer, 'im_end_id'):
            print("为tokenizer添加必要的特殊token ID")
            # 为ChatML格式添加特殊token
            if '<|im_start|>' not in self.tokenizer.get_vocab():
                print("添加<|im_start|>和<|im_end|>到词汇表")
                self.tokenizer.add_special_tokens({
                    'additional_special_tokens': ['<|im_start|>', '<|im_end|>']
                })

            # 手动设置ID
            vocab = self.tokenizer.get_vocab()
            self.tokenizer.im_start_id = vocab.get('<|im_start|>', self.tokenizer.eos_token_id)
            self.tokenizer.im_end_id = vocab.get('<|im_end|>', self.tokenizer.eos_token_id)

        if hasattr(self.generation_config, 'eos_token_id'):
            self.tokenizer.eos_token_id = self.generation_config.eos_token_id

        self.stop_words_ids = []  # 初始化停止词的token ID列表

        from vllm import LLM
        import vllm
        self.__vllm_support_repetition_penalty = version.parse(vllm.__version__) >= version.parse("0.2.2")
        print(f"vLLM版本: {vllm.__version__}")

        quantization = kwargs.get('quantization', None)  # 获取量化参数

        # 初始化模型
        self.model = LLM(model=model_dir,
                         tokenizer=model_dir,
                         tensor_parallel_size=tensor_parallel_size,
                         trust_remote_code=trust_remote_code,
                         quantization=quantization,
                         gpu_memory_utilization=gpu_memory_utilization,
                         dtype=dtype,
                         max_model_len=max_model_len,
                         )

        # 获取停止词的token ID并添加到停止词列表中
        try:
            for stop_id in get_stop_words_ids(self.generation_config.chat_format, self.tokenizer):
                self.stop_words_ids.extend(stop_id)
            if hasattr(self.generation_config, 'eos_token_id'):
                self.stop_words_ids.append(self.generation_config.eos_token_id)
        except Exception as e:
            print(f"设置停止词时出错: {e}")
            # 使用基本停止词
            self.stop_words_ids = [self.tokenizer.eos_token_id] if hasattr(self.tokenizer, 'eos_token_id') else []
            print(f"使用基本停止词: {self.stop_words_ids}")

    def chat(self,
             query: str,
             history: Optional[HistoryType] = None,
             tokenizer: PreTrainedTokenizer = None,
             system: str = "You are a helpful assistant.",
             generation_config: Optional[GenerationConfig] = None,
             **kwargs):
        """进行多轮对话"""
        generation_config = generation_config if generation_config is not None else self.generation_config
        tokenizer = self.tokenizer if tokenizer is None else tokenizer

        # 不再严格检查chat_format，以便适应更多模型
        # assert generation_config.chat_format == 'chatml', _ERROR_BAD_CHAT_FORMAT

        # 检查是否支持重复惩罚
        if not self.__vllm_support_repetition_penalty and hasattr(generation_config, 'repetition_penalty') and generation_config.repetition_penalty != 1:
            print("警告: 当前vLLM版本不支持repetition_penalty，将忽略该参数")

        if history is None:
            history = []
        else:
            # 深拷贝用户输入的历史记录，避免修改原始数据
            history = copy.deepcopy(history)

        max_window_size = kwargs.get('max_window_size', None)
        if max_window_size is None and hasattr(generation_config, 'max_window_size'):
            max_window_size = generation_config.max_window_size
        elif max_window_size is None:
            max_window_size = 6144

        from vllm.sampling_params import SamplingParams

        # 准备基本参数
        sampling_kwargs = {
            "stop_token_ids": self.stop_words_ids,
            "temperature": generation_config.temperature if hasattr(generation_config, 'temperature') else 0.7,
            "max_tokens": generation_config.max_new_tokens if hasattr(generation_config, 'max_new_tokens') else 512,
        }

        # 添加可选参数
        if hasattr(generation_config, 'top_p'):
            sampling_kwargs["top_p"] = generation_config.top_p

        if hasattr(generation_config, 'top_k') and generation_config.top_k != 0:
            sampling_kwargs["top_k"] = -1 if generation_config.top_k == 0 else generation_config.top_k

        # 仅在支持的情况下添加repetition_penalty
        if self.__vllm_support_repetition_penalty and hasattr(generation_config, 'repetition_penalty'):
            sampling_kwargs["repetition_penalty"] = generation_config.repetition_penalty

        # 尝试创建采样参数
        try:
            sampling_params = SamplingParams(**sampling_kwargs)
        except TypeError as e:
            print(f"参数错误: {e}")
            print("尝试移除不支持的参数...")

            # 尝试删除可能不支持的参数
            keys_to_try_remove = ["repetition_penalty", "top_k", "top_p"]
            success = False

            for key in keys_to_try_remove:
                if key in sampling_kwargs:
                    print(f"尝试移除 {key}")
                    tmp_kwargs = sampling_kwargs.copy()
                    tmp_kwargs.pop(key)
                    try:
                        sampling_params = SamplingParams(**tmp_kwargs)
                        sampling_kwargs = tmp_kwargs
                        print(f"成功移除 {key}")
                        success = True
                        break
                    except TypeError:
                        print(f"移除 {key} 后仍有错误")

            if not success:
                print("使用最小参数集")
                sampling_params = SamplingParams(
                    temperature=sampling_kwargs.get("temperature", 0.7),
                    max_tokens=sampling_kwargs.get("max_tokens", 512)
                )

        # 构建上下文
        try:
            raw_text, context_tokens = make_context(
                tokenizer,
                query,
                history=history,
                system=system,
                max_window_size=max_window_size,
                chat_format=generation_config.chat_format if hasattr(generation_config, 'chat_format') else "chatml",
            )
        except Exception as e:
            print(f"使用make_context构建上下文失败: {e}")
            print("尝试直接使用DeepSeek格式...")
            raw_text, context_tokens = make_deepseek_context(tokenizer, query, history, system)

        # 生成回复
        try:
            req_outputs = self.model.generate([query],
                                              sampling_params=sampling_params,
                                              prompt_token_ids=[context_tokens])
            req_output = req_outputs[0]

            prompt_str = req_output.prompt
            prompt_ids = req_output.prompt_token_ids
            req_sample_output_strs = []

            for sample in req_output.outputs:
                output_str = sample.text

                # 清理输出文本
                if IMEND in output_str:
                    output_str = output_str[:-len(IMEND)]
                if ENDOFTEXT in output_str:
                    output_str = output_str[:-len(ENDOFTEXT)]

                # 防止回复中包含Human:
                if "Human:" in output_str:
                    output_str = output_str.split("Human:")[0].strip()

                req_sample_output_strs.append(prompt_str + output_str)

            assert len(req_sample_output_strs) >= 1
            response = req_sample_output_strs[0][len(prompt_str):].strip()

            # 更新历史记录
            history.append((query, response))
            return response, history

        except Exception as e:
            print(f"生成回复时出错: {e}")
            # 尝试直接使用query提示
            try:
                prompt = f"{system}\n\nHuman: {query}\nAssistant:"
                outputs = self.model.generate(prompt, sampling_params=sampling_params)
                response = outputs[0].outputs[0].text.strip()

                # 防止回复中包含Human:
                if "Human:" in response:
                    response = response.split("Human:")[0].strip()

                history.append((query, response))
                return response, history

            except Exception as e2:
                print(f"备用方法也失败: {e2}")
                response = "抱歉，生成回复时出现错误。请重试或尝试其他问题。"
                history.append((query, response))
                return response, history


if __name__ == '__main__':
    # 主程序入口
    import os
    import time

    start_time = time.time()

    # 可以根据实际环境设置环境变量，例如：
    # os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"  # 设置使用的GPU

    model_dir = '/root/autodl-fs/DeepSeek-R1-Distill-Qwen-7B'  # 指定模型目录路径
    tensor_parallel_size = 2  # 设置张量并行度

    print(f"开始初始化模型...")

    # 初始化vLLMWrapper模型
    try:
        model = vLLMWrapper(
            model_dir,
            tensor_parallel_size=tensor_parallel_size,
        )
        print(f"模型初始化完成，耗时 {time.time() - start_time:.2f} 秒")

        # 测试对话
        print("\n===== 测试对话 =====")
        print("用户: 你好")

        test_start = time.time()
        response, history = model.chat(
            query="你好",
            history=None,
            system="你是一个有用的AI助手，请用中文回答问题。"
        )
        print(f"AI: {response}")
        print(f"回复耗时: {time.time() - test_start:.2f} 秒")

        # 第二次对话
        print("\n用户: 给我讲一个年轻人奋斗创业最终取得成功的故事。")
        test_start = time.time()
        response, history = model.chat(
            query="给我讲一个年轻人奋斗创业最终取得成功的故事。",
            history=history
        )
        print(f"AI: {response}")
        print(f"回复耗时: {time.time() - test_start:.2f} 秒")

        # 第三次对话
        print("\n用户: 给这个故事起一个标题")
        test_start = time.time()
        response, history = model.chat(
            query="给这个故事起一个标题",
            history=history
        )
        print(f"AI: {response}")
        print(f"回复耗时: {time.time() - test_start:.2f} 秒")

    except Exception as e:
        print(f"发生错误: {e}")
        import traceback

        traceback.print_exc()
