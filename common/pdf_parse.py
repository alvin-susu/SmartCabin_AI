from typing import List

import pdfplumber
from PyPDF2 import PdfReader

"""
pdf 解析方法

项目最终采用了三种解析方案的综合：
● pdf分块解析，尽量保证一个小标题+对应文档在一个文档块，其中文档块的长度分别是512和1024。
● pdf滑窗法解析，把文档句号分割，然后构建滑动窗口，其中文档块的长度分别是256和512。
● pdf非滑窗法解析，把文档句号分割，然后按照文档块预设尺寸均匀切分，其中文档块的长度分别是256和512。
按照3中解析方案对数据处理之后，然后对文档块做了一个去重，最后把这些文档块输入给召回模块。
"""


class PdfParser:
    def __init__(self, path):
        self.pdf_path = path
        self.context = []

    def get_header(self, page):
        """
        根据页数获取header
        :param page: 页数
        :return: header
        """
        header_dict = self.get_all_header()
        for header in header_dict:
            if header.get('page_number') == page.page_number:
                return header.get('title')

        return None

    def get_all_header(self) -> List[dict]:
        """
        获取大纲目录
        :return: ex.:{'title': '前言 ', 'page_number': 9}
        """
        reader = PdfReader(self.pdf_path)
        outline_data = []
        try:
            outlines = reader.outline
            for item in outlines:
                if isinstance(item, list):
                    for subitem in item:
                        outline_data.append({
                            "title": subitem.title if hasattr(subitem, "title") else str(subitem),
                            "page_number": reader.get_destination_page_number(subitem)
                        })
                else:
                    outline_data.append({
                        "title": item.title if hasattr(item, 'title') else str(item),
                        "page_number": reader.get_destination_page_number(item)
                    })
        except Exception as e:
            print(f"Error reading outline: {e}")

        return outline_data

    def data_filter(self, sequence, max_seq):
        """
        过滤并组织文档块。
        :param sequence: 当前的文档块字符串。
        :param max_seq: 最大序列长度
        """
        # 清理文本 去掉不需要的字符
        sequence = sequence.strip()

        # 如果文档块过短，跳过处理
        if len(sequence) < 10:
            return

        blocks = []

        while len(sequence) > max_seq:
            # 找到前 max_seq 个字符的最后一个换行符或空格，尽量按自然段切分
            split_index = max(
                sequence.rfind("\n", 0, max_seq),
                sequence.rfind(" ", 0, max_seq)
            )

            # 取出当前块并加入块列表
            blocks.append(sequence[:split_index].strip())
            sequence = sequence[split_index:].strip()

        # 添加最后剩余的内容
        if sequence:
            blocks.append(sequence)
        # print(f"保存的当前块内容为:{blocks}")
        self.context.extend(blocks)

    def parse_block(self, max_seq):
        with pdfplumber.open(self.pdf_path) as pdf:
            for page_index, page in enumerate(pdf.pages):
                # 根据页数获取其所在目录
                header = self.get_header(page)
                if header is None:
                    continue
                # 将页面中的每行代码进行分块 并计算它的位置信息
                # use_text_flow 按照文字顺序来加载
                # extra_attrs 除了提取基本信息还会提取字体大小信息
                texts = page.extract_words(use_text_flow=True, extra_attrs=["size"])

                sequence = ""
                last_size = 0
                for line_index, line in enumerate(texts):
                    if line_index < 1:
                        continue
                    if line_index == 1:
                        if line["text"].isdigit():
                            continue
                    cur_size = line["size"]
                    text = line["text"]
                    if text == "□" or text == "•":
                        continue
                    elif text == "警告！" or text == "注意！" or text == "说明！":
                        if len(sequence) > 0:
                            self.data_filter(sequence, max_seq)
                        sequence = ""
                    elif format(last_size, ".5f") == format(cur_size, ".5f"):
                        if len(sequence) > 0:
                            sequence = sequence + text
                        else:
                            sequence = text
                    else:
                        last_size = cur_size
                        if 15 > len(sequence) > 0:
                            sequence = sequence + text
                        else:
                            if len(sequence) > 0:
                                self.data_filter(sequence, max_seq)
                            sequence = text
                if len(sequence) > 0:
                    self.data_filter(sequence, max_seq)

    def parse_sliding_window(self, max_seq=512, min_len=6):
        """
        pdf滑窗法解析，把文档句号分割，然后构建滑动窗口，其中文档块的长度分别是256和512。
        大体逻辑为：先拼接字符串到最大窗口长度，记录该块文字，后剔除头部，添加尾部
        """
        # 加载文档
        with pdfplumber.open(self.pdf_path) as pdf:
            all_content = ""
            # 按页对pdf进行循环处理
            for page_index, page in enumerate(pdf.pages):
                page_content = ""
                # 按顺序提取每一页的文字
                text = page.extract_text(use_text_flow=True)
                # 将每一页的文字以换行符切割
                words = text.split("\n")
                # print(f"未处理前：第{page_index}页文字为:{words}")
                # 循环该页文字
                for word_index, word in enumerate(words):
                    word = word.strip().strip("\n")
                    # 这些情况属于特殊情况 不追加在该页文字中
                    filter_word = ("...................." in word) or ("目录" in word) or (len(word) < 1) or (
                        word.isdigit())
                    if filter_word:
                        continue
                    page_content += word
                    # 字数不够继续拼接文字
                    if len(page_content) < min_len:
                        continue
                    # all_content为了记录pdf的所有文字
                    all_content += page_content
                # print(f"经过字符处理后：第{page_index}页文字为:{page_content}")

            # 将所有文字通过句号分割
            sentences = all_content.split("。")
            # 滑动窗口
            sentences_len = len(sentences)
            fast, slow = 0, 0
            cur = ""
            # 最大块长度
            kernel = max_seq
            while fast < sentences_len:
                sentence = sentences[fast]
                if len(cur + sentence) > kernel and (cur + sentence) not in self.context:
                    # 记录该滑动窗口内的块数据
                    self.context.append(cur + sentence + "。")
                    # 前后衔接 滑动窗口长度为 len(sentences[slow] + "。") 到 末尾，排除上一句，添加下一句
                    cur = cur[len(sentences[slow] + "。"):]
                    # print(f"第{len(self.context)}块数据为:{cur}")
                    slow = slow + 1
                cur = cur + sentence + "。"
                fast = fast + 1

    def parse_not_sliding_window(self, max_seq=512, min_len=6):
        """
        pdf非滑窗法解析，把文档句号分割，然后利用最大长度划分文档块
        :return:
        """
        with pdfplumber.open(self.pdf_path) as pdf:
            for idx, page in enumerate(pdf.pages):
                page_content = ""
                text = page.extract_text()
                words = text.split("\n")
                for word_index, word in enumerate(words):
                    text = word.strip().strip("\n")
                    if "...................." in text or "目录" in text or len(text) < 1 or text.isdigit():
                        continue
                    page_content = page_content + text
                if len(page_content) < min_len:
                    continue
                if len(page_content) < max_seq:
                    if page_content not in self.context:
                        self.context.append(page_content)
                        # print(f"以句号分割的当前块内容为:{page_content}")
                else:
                    sentences = page_content.split("。")
                    cur = ""
                    for sentence_index, sentence in enumerate(sentences):
                        if len(cur + sentence) > max_seq and (cur + sentence) not in self.context:
                            # print(f"以句号分割的当前块内容为:{cur + sentence}")
                            self.context.append(cur + sentence)
                            cur = sentence
                        else:
                            cur = cur + sentence


if __name__ == "__main__":
    pdf_path = "../knowledge_data/pdf/train_a.pdf"
    pdf_parse = PdfParser(pdf_path)
    # pdf_parse.parse_block(max_seq=1024)
    # pdf_parse.parse_block(max_seq=512)
    # print("固定分块的方式解析手册的总块数为{}".format(len(pdf_parse.context)))
    # pdf_parse.parse_sliding_window(max_seq=256)
    # pdf_parse.parse_sliding_window(max_seq=512)
    # print("滑动窗口的方式解析手册的总块数为:{}".format(len(pdf_parse.context)))
    pdf_parse.parse_not_sliding_window(max_seq=256)
    print("非滑动窗口的方式以句号分割解析手册的总块数为:{}".format(len(pdf_parse.context)))
    pdf_parse.parse_not_sliding_window(max_seq=512)
    print("非滑动窗口的方式以句号分割解析手册的总块数为:{}".format(len(pdf_parse.context)))
    # data = dp.data
    # out = open("all_text.txt", "w")
    # for line in data:
    #     line = line.strip("\n")
    # out.write(line)
    # out.write("\n")
    # out.close()
