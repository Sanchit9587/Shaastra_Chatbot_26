# --- 1. CRITICAL FIX FOR CHROMA DB ---
try:
    __import__('pysqlite3')
    import sys
    sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')
except ImportError:
    pass
# -------------------------------------

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, pipeline
import sys
from typing import Union, List, Optional, Any, Dict

# --- LANGCHAIN IMPORTS ---
from langchain.prompts import PromptTemplate
from langchain_community.document_loaders import TextLoader
from langchain_core.caches import BaseCache
from langchain_core.callbacks import Callbacks, BaseCallbackHandler
from langchain_core.language_models import BaseLanguageModel

# --- RETRIEVAL IMPORTS ---
from langchain.retrievers import ParentDocumentRetriever, EnsembleRetriever
from langchain_community.retrievers import BM25Retriever
from langchain.storage import InMemoryStore
from langchain_text_splitters import RecursiveCharacterTextSplitter, MarkdownHeaderTextSplitter
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings, HuggingFacePipeline

# --- CRITICAL PYDANTIC FIX ---
try:
    HuggingFacePipeline.model_rebuild(_types_namespace=globals())
except Exception:
    pass 

def check_gpu_status():
    if torch.cuda.is_available():
        return 'cuda'
    return 'cpu'

# --- CONFIGURATION ---
LLM_CONFIG = {
    "Unsloth 3B": {
        "model_id": "unsloth/llama-3.2-3b-instruct-bnb-4bit",
        "prompt_template": """<|begin_of_text|><|start_header_id|>system<|end_header_id|>

You are a helpful assistant for Shaastra 2025.
Use ONLY the context provided below.
If the context contains a Schedule Table, look strictly at the row corresponding to the Event requested.
If there are conflicting details, prioritize the 'Event Details' section over the 'Schedule'.

<|eot_id|><|start_header_id|>user<|end_header_id|>

CONTEXT:
{context}

QUESTION: {question}<|eot_id|><|start_header_id|>assistant<|end_header_id|>
"""
    }
}

CONTEXT_FILE = "rag_context.md"

# --- 2. HYBRID RETRIEVER SETUP ---

def create_hybrid_retriever(file_path, embedding_model):
    print(f"Loading document from {file_path}...")
    loader = TextLoader(file_path)
    docs = loader.load()

    # --- A. PREPARE CHUNKS ---
    # 1. Markdown Splitter (Creates the logical Parent Chunks)
    headers_to_split_on = [
        ("#", "Header 1"),
        ("##", "Header 2"),
        ("###", "Header 3"),
    ]
    md_splitter = MarkdownHeaderTextSplitter(headers_to_split_on=headers_to_split_on)
    parent_docs = md_splitter.split_text(docs[0].page_content)

    # 2. Child Splitter (Small chunks for vector search)
    child_splitter = RecursiveCharacterTextSplitter(chunk_size=400, chunk_overlap=50)

    # --- B. VECTOR RETRIEVER (ParentDocumentRetriever) ---
    vectorstore = Chroma(
        collection_name="shaastra_hierarchical", 
        embedding_function=embedding_model,
        persist_directory="./chroma_db_hierarchical" 
    )
    store = InMemoryStore()

    print("1. Creating Vector Retriever (Hierarchical)...")
    vector_retriever = ParentDocumentRetriever(
        vectorstore=vectorstore,
        docstore=store,
        child_splitter=child_splitter,
        parent_splitter=RecursiveCharacterTextSplitter(chunk_size=2000),
        search_kwargs={"k": 4} 
    )
    
    print("   Indexing Vector Docs...")
    vector_retriever.add_documents(parent_docs, ids=None)

    # --- C. KEYWORD RETRIEVER (BM25) ---
    print("2. Creating Keyword Retriever (BM25)...")
    # We feed BM25 the Parent Docs directly so it returns full context on keyword match
    bm25_retriever = BM25Retriever.from_documents(parent_docs)
    bm25_retriever.k = 2  # Retrieve top 2 exact keyword matches

    # --- D. ENSEMBLE (HYBRID) ---
    print("3. combining into Hybrid Ensemble...")
    ensemble_retriever = EnsembleRetriever(
        retrievers=[bm25_retriever, vector_retriever],
        weights=[0.4, 0.6] # 40% priority to keywords, 60% to semantic meaning
    )
    
    print("Hybrid Retrieval System Ready.")
    return ensemble_retriever

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
        max_new_tokens=512,
        return_full_text=False
    )
    
    return HuggingFacePipeline(pipeline=hf_pipeline)