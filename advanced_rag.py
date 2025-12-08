# --- 1. CRITICAL FIX FOR CHROMA DB (SQLite FTS5) ---
# This block must remain at the very top before any other imports
try:
    __import__('pysqlite3')
    import sys
    sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')
except ImportError:
    pass
# ---------------------------------------------------

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, pipeline
import sys

# --- TYPING IMPORTS (Required for Pydantic Fix) ---
from typing import Union, List, Optional, Any, Dict, Iterator, AsyncIterator

# --- LANGCHAIN IMPORTS ---
from langchain.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_community.document_loaders import TextLoader

# ** IMPORTANT: These imports are required for HuggingFacePipeline to rebuild correctly **
from langchain_core.caches import BaseCache
from langchain_core.callbacks import Callbacks, BaseCallbackHandler
from langchain_core.language_models import BaseLanguageModel

# --- HIERARCHICAL & VECTOR STORE IMPORTS ---
from langchain.retrievers import ParentDocumentRetriever
from langchain.storage import InMemoryStore
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings, HuggingFacePipeline

# --- CRITICAL PYDANTIC FIX ---
# Forces Pydantic to rebuild the schema with all necessary types visible in the global scope
try:
    HuggingFacePipeline.model_rebuild(_types_namespace=globals())
except Exception as e:
    pass 
# -----------------------------

# --- CONFIGURATION ---
LLM_CONFIG = {
    "Unsloth 3B (Cached)": {
        "model_id": "unsloth/llama-3.2-3b-instruct-bnb-4bit",
        "prompt_template": """<|begin_of_text|><|start_header_id|>system<|end_header_id|>

You are a helpful and precise assistant for the Shaastra 2025 tech festival.
You must answer the user's question using ONLY the context provided.
If there are multiple dates, venues, or rounds (e.g., Prelims vs Finals), CLEARLY distinguish between them.
If the answer is not in the context, say "I do not have enough information."<|eot_id|><|start_header_id|>user<|end_header_id|>

CONTEXT:
---
{context}
---

QUESTION: {question}<|eot_id|><|start_header_id|>assistant<|end_header_id|>
"""
    }
}

EMBEDDING_MODEL_ID = 'all-MiniLM-L6-v2'
CONTEXT_FILE = "rag_context.md"

# --- GPU CHECK ---
def check_gpu_status():
    if torch.cuda.is_available():
        print(f"✅ GPU Detected: {torch.cuda.get_device_name(0)}")
        return 'cuda'
    print("❌ No GPU detected. Running on CPU.")
    return 'cpu'

# --- 2. HIERARCHICAL RETRIEVER SETUP ---

def create_hierarchical_retriever(file_path, embedding_model):
    """
    Creates a ParentDocumentRetriever.
    It indexes small 'Child' chunks for precise search, but returns large 'Parent' chunks
    to the LLM so it has full context (solving the Moot Court issue).
    """
    print(f"Loading document from {file_path}...")
    loader = TextLoader(file_path)
    docs = loader.load()

    # 1. Define Splitters
    # Parent: Large chunks (The full context the LLM will see)
    # 2000 chars is enough to cover a full event description table or section
    parent_splitter = RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=200)
    
    # Child: Small chunks (For precise searching)
    # 400 chars is small enough to match specific keywords like "Finals"
    child_splitter = RecursiveCharacterTextSplitter(chunk_size=400, chunk_overlap=50)

    # 2. Vectorstore (Stores the Child chunks for search)
    vectorstore = Chroma(
        collection_name="hierarchical_parents", 
        embedding_function=embedding_model,
        persist_directory="./chroma_db_hierarchical"
    )

    # 3. Docstore (Stores the Parent chunks for retrieval)
    store = InMemoryStore()

    # 4. Initialize Retriever
    print("Creating Hierarchical Retriever (ParentDocumentRetriever)...")
    retriever = ParentDocumentRetriever(
        vectorstore=vectorstore,
        docstore=store,
        child_splitter=child_splitter,
        parent_splitter=parent_splitter,
    )

    # 5. Add documents (This splits and indexes them)
    # Note: Only re-index if needed. For now, we do it every run for simplicity.
    print("Indexing documents (this creates parent & child chunks)...")
    retriever.add_documents(docs)
    print("Indexing complete.")
    
    return retriever

# --- 3. LLM LOADING ---

def load_llm(model_id):
    print(f"Loading LLM: {model_id}...")
    check_gpu_status()

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
    )
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        quantization_config=bnb_config,
        device_map="auto",
        torch_dtype=torch.float16,
        trust_remote_code=True
    )
    hf_pipeline = pipeline(
        "text-generation", 
        model=model, 
        tokenizer=tokenizer, 
        max_new_tokens=500, # Increased tokens for detailed answers
        return_full_text=False
    )
    
    return HuggingFacePipeline(pipeline=hf_pipeline)