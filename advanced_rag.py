# --- 1. CRITICAL FIX FOR CHROMA DB (SQLite FTS5) ---
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

# --- TYPING IMPORTS ---
from typing import Union, List, Optional, Any, Dict, Iterator, AsyncIterator

# --- LANGCHAIN IMPORTS ---
from langchain.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_community.document_loaders import TextLoader

# --- TYPES NEEDED FOR PYDANTIC REBUILD ---
from langchain_core.caches import BaseCache
from langchain_core.callbacks import Callbacks, BaseCallbackHandler
from langchain_core.language_models import BaseLanguageModel

# --- RETRIEVAL IMPORTS ---
from langchain.retrievers import ParentDocumentRetriever
from langchain.storage import InMemoryStore
from langchain_text_splitters import RecursiveCharacterTextSplitter, MarkdownHeaderTextSplitter
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings, HuggingFacePipeline

# --- CRITICAL PYDANTIC FIX ---
# We explicitely define the namespace so Pydantic finds 'Union', 'Callbacks', etc.
try:
    HuggingFacePipeline.model_rebuild(_types_namespace={
        "Union": Union,
        "List": List,
        "Optional": Optional,
        "Dict": Dict,
        "Any": Any,
        "Callbacks": Callbacks,
        "BaseCallbackHandler": BaseCallbackHandler,
        "BaseCache": BaseCache,
        "BaseLanguageModel": BaseLanguageModel
    })
except Exception as e:
    print(f"Pydantic Rebuild Warning: {e}")
# -----------------------------

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
{system_instruction}
<|eot_id|><|start_header_id|>user<|end_header_id|>

CONTEXT:
{context}

QUESTION: {question}<|eot_id|><|start_header_id|>assistant<|end_header_id|>
"""
    }
}

CONTEXT_FILE = "rag_context.md"

# --- 2. HIERARCHICAL RETRIEVER (Markdown Aware) ---

def create_hierarchical_retriever(file_path, embedding_model):
    print(f"Loading document from {file_path}...")
    loader = TextLoader(file_path)
    docs = loader.load()

    # 1. Markdown Splitter (Keeps "Day 1" header attached to "Events")
    headers_to_split_on = [
        ("#", "Header 1"),
        ("##", "Header 2"),
        ("###", "Header 3"),
    ]
    md_splitter = MarkdownHeaderTextSplitter(headers_to_split_on=headers_to_split_on)
    # These are our "Parent" documents
    parent_docs = md_splitter.split_text(docs[0].page_content)

    # 2. Child Splitter (Small chunks for vector search)
    child_splitter = RecursiveCharacterTextSplitter(chunk_size=400, chunk_overlap=50)

    # 3. Vectorstore (Indexes Children)
    vectorstore = Chroma(
        collection_name="shaastra_hierarchical", 
        embedding_function=embedding_model,
        persist_directory="./chroma_db_hierarchical" 
    )

    # 4. Docstore (Stores Parents)
    store = InMemoryStore()

    print("Creating Hierarchical Retriever...")
    retriever = ParentDocumentRetriever(
        vectorstore=vectorstore,
        docstore=store,
        child_splitter=child_splitter,
        # We manually feed parent docs, so we don't need a parent_splitter here
        # but the class requires the argument or specific usage.
        # We will use the add_documents method which handles splitting parents into children.
        parent_splitter=RecursiveCharacterTextSplitter(chunk_size=2000) # Fallback
    )

    print(f"Indexing {len(parent_docs)} parent sections...")
    retriever.add_documents(parent_docs, ids=None)
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
        max_new_tokens=512,
        return_full_text=False
    )
    
    return HuggingFacePipeline(pipeline=hf_pipeline)