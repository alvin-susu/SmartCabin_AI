import time

from langchain.chains.retrieval_qa.base import RetrievalQA
from langchain_core.prompts import PromptTemplate

from common.pdf_parse import PdfParser
from main.bm25_retriever import BestMatch25Retriever
from main.faiss_retriever import FaissRetriever
from main.vllm_model import ChatLLM


# 获取langchain的工具链
def get_qa_chain(llm, vector_store, prompt_template):
    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    return RetrievalQA.from_llm(llm=llm, retriever=vector_store.as_retriever(search_kwargs={"k": 10}), prompt=prompt)


def get_emb_bm25_merge(faiss_context, bm25_context, query):
    max_length = 2500
    # faiss向量库匹配结果处理
    emb_ans = ""
    cnt = 0
    for doc, score in faiss_context:
        cnt = cnt + 1
        if len(emb_ans + doc.page_content) > max_length:
            break

        # 最多取前5个
        if cnt > 6:
            break
        emb_ans = emb_ans + doc.page_content

    # bm25相似性匹配的结果处理
    bm25_ans = ""
    cnt = 0
    for doc in bm25_context:
        cnt = cnt + 1
        if len(bm25_ans + doc.page_content) > max_length:
            break
        bm25_ans = bm25_ans + doc.page_content
        # 最多取前5个
        if cnt > 6:
            break

    prompt_template = """基于以下已知信息，简洁和专业的来回答用户的问题。
                                如果无法从中得到答案，请说 "无答案"或"无答案"，不允许在答案中添加编造成分，答案请使用中文。
                                已知内容为吉利控股集团汽车销售有限公司的吉利用户手册:
                                1: {emb_ans}
                                2: {bm25_ans}
                                问题:
                                {question}""".format(emb_ans=emb_ans, bm25_ans = bm25_ans, question = query)
    return prompt_template

def get_rerank(emb_ans, query):
    prompt_template = """基于以下已知信息，简洁和专业的来回答用户的问题。
                                如果无法从中得到答案，请说 "无答案"或"无答案" ，不允许在答案中添加编造成分，答案请使用中文。
                                已知内容为吉利控股集团汽车销售有限公司的吉利用户手册:
                                1: {emb_ans}
                                问题:
                                {question}""".format(emb_ans=emb_ans, question = query)
    return prompt_template

def question(text, llm, vector_store, prompt_template):
    chain = get_qa_chain(llm, vector_store, prompt_template)
    response = chain({"query": text})
    return response


def rerank(re_rank, top_k, query, bm25_ans, faiss_ans):
    items = []
    max_length = 4000
    for doc, score in faiss_ans:
        items.append(doc)
    items.extend(bm25_ans)
    re_rank_ans = re_rank.predict(query, items)
    re_rank_ans = re_rank_ans[:top_k]

    emb_ans = ""
    for doc in re_rank_ans:
        if len(emb_ans + doc.page_content) > max_length:
            break
        emb_ans = emb_ans + doc.page_content
    return emb_ans

if __name__ == "__main__":
    start = time.time()

    qwen7 = "../pre_trained_models/Qwen/Qwen-7B-Chat"
    m3e = "../pre_trained_models/moka-ai/m3e-base"
    bge_reranker_large = "../pre_trained_models/BAAI/bge-reranker-large"
    pdf_path = "../knowledge_data/pdf/train_a.pdf"

    # 解析pdf文档，构造数据
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
    # Faiss召回
    faiss_retriever = FaissRetriever(pdf_path)
    vector_store = faiss_retriever.get_vector()
    print("faiss_retriever load ok")
    # BM25召回
    bm25_retriever = BestMatch25Retriever(pdf_path)
    print("bm25 load ok")

    # LLM大模型
    llm = ChatLLM(qwen7)
    print("llm qwen load ok")

    # reRank模型
    rerank = reRankLLM(bge_reranker_large)
    print("rerank model load ok")

    # 对每一条测试问题做答案生成处理