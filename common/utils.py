"""
通用工具
"""
import os
from typing import List, Tuple

from transformers import PreTrainedTokenizer


def load_sub_files(parent_path: str):
    """
    加载路径下的所有文件并返回绝对路径
    :param parent_path: 父目录
    :return: 绝对路径
    """

    files_list = []
    for root, dirs, files in os.walk(parent_path):
        abspath = os.path.abspath(root)
        all_path = os.path.join(abspath, *files)
        print(f"all_path: {all_path}")
        files_list.append(all_path)
    return files_list


def make_context(
        tokenizer: PreTrainedTokenizer,
        query: str,
        history: List[Tuple[str, str]] = None,
        system: str = "",
        max_window_size: int = 6144,
        chat_format: str = "chatml",
):
    """
    @param tokenizer: 分词器对象，用于将文本转换为 token ID。
    @param query: 当前用户问题
    @param history: 对话历史，列表形式，每个元素是一个 (用户问题, 模型回答) 的元组。
    @param system: 系统设定文本
    @param max_window_size: 上下文的最大长度（以 token 为单位）。
    @param chat_format: 聊天格式，目前支持 "chatml"。
    @return:
    """
    if history is None:
        history = []
    if chat_format == "chatml":
        im_start, im_end = "<|im_start|>", "<|im_end|>"
        # 分词器的开始token
        im_start_tokens = [tokenizer.im_start_id]
        # 分词器的结束token
        im_end_tokens = [tokenizer.im_end_id]
        # 还行符的token
        nl_tokens = tokenizer.encode("\n")

        # 定义一个分词器的
        def _tokenize_str(role, content):
            """
            将角色和内容组合成一个字符串，并将其转换为 token ID 序列。
            参数:
                @param role: 角色名称（如 "user" 或 "assistant"）。
                @param content: 内容文本。

            返回:
                - 组合后的字符串。
                - 组合后的 token ID 序列。
            """
            return f"{role}\n{content}", tokenizer.encode(role, allowed_special=set()) + nl_tokens + tokenizer.encode(
                content, allowed_special=set())

        # 传递参数 system和定义system的设定
        system_text, system_tokens_part = _tokenize_str("system", system)
        # 为encode编码之后的系统设定来增加前后分割符
        system_tokens = im_start_tokens + system_tokens_part + im_end_tokens

        # 处理对话历史
        # 从最近的对话开始，逐步向前遍历对话历史
        raw_text = ""
        context_tokens = []
        for turn_query, turn_response in reversed(history):

            # 处理用户问题
            query_text, query_tokens_part = _tokenize_str("user", turn_query)
            query_tokens = im_start_tokens + query_tokens_part + im_end_tokens

            # 处理模型回答
            response_text, response_tokens_part = _tokenize_str("assistant", turn_response)
            response_tokens = im_start_tokens + response_tokens_part + im_end_tokens

            # 组合当前轮次的上下文
            next_context_tokens = nl_tokens + query_tokens + nl_tokens + response_tokens
            prev_chat = f"\n{im_start}{query_text}{im_end}\n{im_start}{response_text}{im_end}"

            # 检查上下文是否超出限制
            current_context_size = (len(system_tokens) + len(next_context_tokens) + len(context_tokens))
            if current_context_size < max_window_size:
                context_tokens = next_context_tokens + context_tokens
                raw_text = prev_chat + raw_text
            else:
                break
        # 处理系统设定+上下文
        context_tokens = system_tokens + context_tokens
        raw_text = f"{im_start}{system_text}{im_end}" + raw_text

        # 处理当前问题
        context_tokens += (
                nl_tokens + im_start_tokens + _tokenize_str("user", query)[1] + im_end_tokens
                + nl_tokens + im_start_tokens + tokenizer.encode("assistant") + nl_tokens
        )
        raw_text += f"\n{im_start}user\n{query}{im_end}\n{im_start}assistant\n"

    elif chat_format == "raw":
        raw_text = query
        context_tokens = tokenizer.encode(raw_text)
    else:
        raise NotImplementedError(f"Unknown chat format {chat_format!r}")

    return raw_text, context_tokens


if __name__ == "__main__":
    print(load_sub_files("../knowledge_data"))
