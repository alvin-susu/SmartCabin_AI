"""
BM25 只考虑字面相似性，对深层语义检索无能为力。
适合与向量检索（如 FAISS）结合使用，形成混合检索系统。
可以将 BM25 检索的结果作为初筛，再结合向量检索方法（如 FAISS）进行深度检索
"""
import jieba
from langchain.schema import Document
from langchain_community.retrievers import BM25Retriever

from common.pdf_parse import PdfParser


class BestMatch25Retriever(object):

    # 遍历文档 首先做分词 然后把分词后的文档和全文文档建立索引和映射关系
    def __init__(self, documents):
        docs = []
        full_docs = []
        for idx, line in enumerate(documents):
            # 去除多余字符
            line = line.strip("\n").strip()
            if len(line) < 5:
                continue

            # jieba 是一个流行的中文分词库，支持多种分词模式
            # cut_for_search 是 jieba 的一种分词模式，专门用于搜索引擎场景
            # 例如，"清华大学" 会被切分为 ["清华", "大学", "清华大学"]
            # 最终 tokens 的值为 "我 爱 北京 天安门 天门"
            tokens = " ".join(jieba.cut_for_search(line))
            # 拼接分词后的tokens 然后设置meta_data的键为id
            docs.append(Document(page_content=tokens, meta_data={"id": idx}))
            # 原文分词
            words = line.split("\t")
            # 原文构建索引
            full_docs.append(Document(page_content=words[0], meta_data={"id": idx}))

        # 分词后的文档列表
        self.documents = docs
        # 原始文档的分词列表
        self.full_documents = full_docs
        # 初始化 BM25 检索器
        self.retriever = self._init_bm25()

    def _init_bm25(self):
        return BM25Retriever.from_documents(self.documents)

    def get_bm25_top_k(self, query: str, top_k: int):
        """
        根据bm25来查询字面相似结果
        @param  query: 查询
        @param top_k: 前k个
        :return: 查询结果
        """
        # 对查询进行分词
        query = " ".join(jieba.cut_for_search(query))
        # 使用BM25检索相关文档
        # FIXME：返回的ans_docs中的metadata为空
        ans_docs = self.retriever.invoke(query, k=top_k)
        print(f"ans_docs:{ans_docs}")
        ans = []
        # 遍历检索结果，获取原始文档
        for line in ans_docs:
            # 检查 metadata 中是否有 id
            # 如果 metadata 为空，手动添加 id
            if not line.metadata:
                # 使用当前结果的索引作为 id
                line.metadata = {"id": len(ans)}
            if "id" in line.metadata:
                # 根据 id 获取原始文档
                ans.append(self.full_documents[line.metadata["id"]])
            else:
                print("Warning: Document metadata is missing 'id'", line.metadata)
        return ans


if __name__ == "__main__":
    # 示例文档
    pdf_path = "../knowledge_data/pdf/train_a.pdf"
    pdf_parser = PdfParser(pdf_path)
    pdf_parser.parse_block(max_seq=1024)
    pdf_parser.parse_block(max_seq=512)
    print(len(pdf_parser.context))
    pdf_parser.parse_sliding_window(max_seq=512)
    pdf_parser.parse_sliding_window(max_seq=256)
    print(len(pdf_parser.context))
    pdf_parser.parse_not_sliding_window(max_seq=512)
    pdf_parser.parse_not_sliding_window(max_seq=256)
    print(len(pdf_parser.context))
    data = pdf_parser.context
    print("data load ok")
    bm25 = BestMatch25Retriever(data)
    res = bm25.get_bm25_top_k("座椅加热", 6)
    print(res)
