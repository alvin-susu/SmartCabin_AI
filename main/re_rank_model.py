import os

import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

from main.config import LLM_DEVICE

os.environ["TOKENIZERS_PARALLELISM"] = "false"

DEVICE = LLM_DEVICE
DEVICE_ID = "0"
CUDA_DEVICE = f"{DEVICE}:{DEVICE_ID}" if DEVICE_ID else DEVICE


# 释放gpu上没有用到的显存以及显存碎片
def torch_gc():
    if torch.cuda.is_available():
        with torch.cuda.device(CUDA_DEVICE):
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()


# 加载rerank模型
class ReRankModel(object):
    def __init__(self, model_path, max_length=512):
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_path)
        # 将模型设置为评估模式
        self.model.eval()
        # 将模型的参数和计算转换为半精度浮点数
        self.model.half()
        # 将模型移动到GPU进行计算
        self.model.cuda()
        self.max_length = max_length

    # 输入文档对，返回每一对（query,doc）的相关得分，并从大到小排序
    def predict(self, query, docs):
        pairs = [(query, doc.page_content) for doc in docs]
        inputs = self.tokenizer(pairs, padding=True, truncation=True, eturn_tensors='pt',
                                max_length=self.max_length).to("cuda")
        # 禁用梯度计算（评估模式下不需要梯度）
        with torch.no_grad():
            # 传入模型评估问题和答案之间的相关性
            scores = self.model(**inputs).numpy()
        scores = scores.detach().cpu().clone().numpy()
        response = [doc for score, doc in sorted(zip(scores, docs), key=lambda x: x[0], reverse=True)]
        # 释放gpu上没有用到的显存以及显存碎片
        torch_gc()
        return response


if __name__ == "__main__":
    bge_reranker_large = "../pre_train_model/BAAI/bge-reranker-large"
    rerank = ReRankModel(bge_reranker_large)
    print(rerank)
