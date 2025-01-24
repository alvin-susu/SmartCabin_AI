from langchain_core.embeddings import Embeddings
from sentence_transformers import SentenceTransformer


class SentenceTransformerEmbeddings(Embeddings):
    def __init__(self):
        self.model = SentenceTransformer("moka-ai/m3e-base")

    def embed_query(self, query: str) -> list:
        """
        将查询转为向量
        :param query:  查询
        :return: 向量
        """
        return self.model.encode(query, show_progress_bar=False, convert_to_numpy=True).tolist()

    def embed_documents(self, documents: list) -> list:
        """
        将文档转为向量
        :param documents:
        :return:
        """
        return self.model.encode(documents, show_progress_bar=False, convert_to_numpy=True).tolist()

