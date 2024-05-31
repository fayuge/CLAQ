
from huggingface_hub import snapshot_download

snapshot_download(repo_id="facebook/opt-125m", allow_patterns=["*.json", "pytorch_model.bin", "vocab.txt"], local_dir="./my_model/")

