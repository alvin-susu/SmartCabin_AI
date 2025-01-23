import os

import torch

EMBEDDING_DEVICE = "cpu"

if torch.cuda.is_available():
    EMBEDDING_DEVICE = "cuda"
if torch.backends.mps.is_available():
    EMBEDDING_DEVICE = "mps"

LLM_DEVICE = "cpu"
if torch.cuda.is_available():
    LLM_DEVICE = "cuda"
if torch.backends.mps.is_available():
    LLM_DEVICE = "mps"

num_gpus = torch.cuda.device_count()

# model cache config
MODEL_CACHE_PATH = os.path.join(os.path.dirname(__file__), 'model_cache')

# vector storage config
VECTOR_STORE_PATH='../vector_store'
COLLECTION_NAME='my_collection'
