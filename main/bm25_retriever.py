"""
BM25 只考虑字面相似性，对深层语义检索无能为力。
适合与向量检索（如 FAISS）结合使用，形成混合检索系统。
可以将 BM25 检索的结果作为初筛，再结合向量检索方法（如 FAISS）进行深度检索
"""

from langchain.schema import Document
from langchain_community.retrievers import BM25Retriever

class BestMatch25Retriever:
    def __init__(self):
        self.retriever = BM25Retriever

    def get_retriever_documents(self, documents, query: str):
        """
        根据bm25来查询字面相似结果
        :param documents:
        :param query: 查询
        :return: 查询结果
        """
        retriever = self.retriever.from_documents(documents)
        return retriever.invoke(query)



if __name__ == "__main__":
    # 示例文档
    documents = [
        Document(page_content="申请成果积分创新成果介绍", metadata={"id": "1"}),
        Document(page_content="文档包含科研项目管理的详细信息", metadata={"id": "2"}),
        Document(page_content="创新积分的申请要求与流程", metadata={"id": "3"})
    ]

    bm_retriever = BestMatch25Retriever()
    results = bm_retriever.get_retriever_documents(documents,  "创新成果申请流程")


    # 初始化 BM25 Retriever
    # retriever = BM25Retriever.from_documents(documents)

    # 查询示例
    # query =
    # results = retriever.get_relevant_documents(query)

    # 打印结果
    for result in results:
        print(f"内容: {result.page_content}, 元数据: {result.metadata}")
