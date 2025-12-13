# embeddings.py
from langchain_huggingface import HuggingFaceEmbeddings
import torch

def get_embedding_model():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Loading embeddings on {device}...")
    return HuggingFaceEmbeddings(
        model_name="BAAI/bge-base-en-v1.5",
        model_kwargs={'device': device}
    )