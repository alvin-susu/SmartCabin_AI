# Load model directly
# from transformers import AutoModelForCausalLM
# model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen-7B-Chat", trust_remote_code=True)
from huggingface_hub import snapshot_download

snapshot_download(repo_id="Qwen/Qwen-7B-Chat",    # 模型ID
                  local_dir="./Qwen-7B-Chat") # 指定本地地址保存模型