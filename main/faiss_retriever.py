import io
import json
import os

import faiss
from langchain_community.docstore.in_memory import InMemoryDocstore
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document

from common.pdf_parse import PdfParser
from pre_train_model.m3e_large import SentenceTransformerEmbeddings

"""
向量召回利用 FAISS 进行索引创建和查找，embedding 利用 M3E-large 。
M3E 是 Moka Massive Mixed Embedding 的缩写
● Moka，此模型由 MokaAI 训练，开源和评测，训练脚本使用 uniem ，评测 BenchMark 使用 MTEB-zh
● Massive，此模型通过千万级 (2200w+) 的中文句对数据集进行训练
● Mixed，此模型支持中英双语的同质文本相似度计算，异质文本检索等功能，未来还会支持代码检索
● Embedding，此模型是文本嵌入模型，可以将自然语言转换成稠密的向量
M3E共有以下三种模型：small, base, large。本项目采用的是M3E large。

数据准备:

收集待检索的文档，并将其处理成适合索引的数据格式。
为文档生成两个版本：
    Embedding 版本: 使用 M3E-large 模型将文本转换为向量。
    文本版本: 原始文档文本。

索引构建:
    FAISS 索引: 使用 M3E-large 提取文档的向量表示，并构建 FAISS 向量索引。
    BM25 索引: 使用 BM25 工具（如 ElasticSearch 或 Whoosh）构建文本倒排索引。

查询处理:
用户输入的查询，处理成两种形式：
    Embedding 查询: 使用 M3E-large 将查询文本转为向量。
    关键词查询: 使用原始查询文本。

双路径检索:
    向量检索（FAISS）:
        使用查询的向量表示在 FAISS 中检索最相似的向量。
    关键词检索（BM25）:
        使用查询文本在 BM25 索引中检索最相关的文档。

结果融合:
将 FAISS 和 BM25 的检索结果融合，可以使用以下策略：
    直接合并: 将两种召回的结果合并，去重，并按某种评分规则排序。
    加权融合: 根据业务需求对两种检索结果赋予权重，计算综合得分并排序。

展示结果:
返回经过融合排序的最终检索结果。

映射关系：在向量检索中，FAISS 索引只存储了向量，而不存储向量对应的原始文档内容。如果没有映射关系，我们就无法从检索结果中找到原始文档。
"""

# index = faiss.IndexFlatL2(len(m3e_embeddings.embed_query("hello world")))
#

# 加载pdf文件
pdfParser = PdfParser('../data/test.pdf')
pdfParser.parse_sliding_window()
context_list = pdfParser.context

# 创建或加载索引
index_bin_name = "faiss_index_m3e.bin"
index_path = "../vector_db"
index_file = os.path.join(index_path, index_bin_name)

if not os.path.exists(index_path):
    os.makedirs(index_path)

# 加载m3e模型
embedding_model = SentenceTransformerEmbeddings()

# 索引文件不存在
if index_bin_name not in os.listdir(index_path):

    # 分词处理 利用m3e-large将加载数据变为稠密向量
    embeddings = [embedding_model.embed_documents(context) for context in context_list]

    # 创建索引
    # 向量的维度
    dimension = len(embeddings[0])
    # 使用L2距离度量
    index = faiss.IndexFlatL2(dimension)

    # 创建向量存储
    vector_store = FAISS(
        embedding_function=embedding_model,
        index=index,
        docstore=InMemoryDocstore(
            {str(i): Document(page_content=context_list[i], metadata={id: i}) for i in range(len(context_list))}
        ),
        index_to_docstore_id={i: str(i) for i in range(len(context_list))},
    )

    # 添加文本和向量到向量存储
    vector_store.add_texts(
        texts=context_list,
        embeddings=embeddings
    )

    faiss.write_index(index, os.path.join(index_path, index_bin_name))
    print("创建索引文件成功！")
else:
    # 索引文件存在
    index = faiss.read_index(os.path.join(index_path, index_bin_name))
    print("加载索引文件成功！")

    # 恢复向量存储
    vector_store = FAISS(
        embedding_function=embedding_model,
        index=index,
        docstore=InMemoryDocstore(
            {str(i): Document(page_content=context_list[i], metadata={id: i}) for i in range(len(context_list))}
        ),
        index_to_docstore_id={i: str(i) for i in range(len(context_list))},
    )

# 查询向量
results = vector_store.similarity_search("申请成果积分创新成果介绍", k=2)
for res in results:
    print(f"* {res.page_content} [{res.metadata}]")
