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
import networkx as nx
from typing import Union, List, Optional, Any, Dict

# --- LANGCHAIN IMPORTS ---
from langchain.prompts import PromptTemplate
from langchain_community.document_loaders import TextLoader
from langchain_core.caches import BaseCache
from langchain_core.callbacks import Callbacks, BaseCallbackHandler
from langchain_core.language_models import BaseLanguageModel
from langchain_core.documents import Document

# --- RETRIEVAL IMPORTS ---
from langchain.retrievers import ParentDocumentRetriever, ContextualCompressionRetriever
from langchain.retrievers.document_compressors import CrossEncoderReranker
from langchain.storage import InMemoryStore
from langchain_text_splitters import RecursiveCharacterTextSplitter, MarkdownHeaderTextSplitter
from langchain_chroma import Chroma
from langchain_community.cross_encoders import HuggingFaceCrossEncoder
from langchain_huggingface import HuggingFaceEmbeddings, HuggingFacePipeline

# --- CRITICAL PYDANTIC FIX ---
try:
    HuggingFacePipeline.model_rebuild(_types_namespace=globals())
except Exception:
    pass 

# --- CONFIGURATION ---
LLM_CONFIG = {
    "Unsloth 3B": {
        "model_id": "unsloth/llama-3.2-3b-instruct-bnb-4bit",
        # CO-T PROMPT WITH MEMORY INJECTION
        "prompt_template": """<|begin_of_text|><|start_header_id|>system<|end_header_id|>

You are an intelligent assistant for Shaastra 2025.
Use the Context and Chat History to answer.

RULES:
1. If the user asks "Where is it?" or "When is it?", look at the CHAT HISTORY to see what they are talking about.
2. If the Context contains conflicting info, prioritize "Event Details" over "Schedule".
3. Think step-by-step.

<|eot_id|><|start_header_id|>user<|end_header_id|>

CHAT HISTORY:
{chat_history}

CONTEXT:
{context}

QUESTION: {question}<|eot_id|><|start_header_id|>assistant<|end_header_id|>
Thinking Process:
1. Identify the core entity (Event/Venue) from the question or history.
2. Cross-reference dates and venues in the context.
3. Formulate the final answer.

Answer:
"""
    }
}

CONTEXT_FILE = "rag_context.md"
RERANKER_MODEL_ID = "cross-encoder/ms-marco-MiniLM-L-6-v2"

def check_gpu_status():
    if torch.cuda.is_available():
        return 'cuda'
    return 'cpu'

# --- 2. KNOWLEDGE GRAPH BUILDER (Simple NetworkX) ---
# This creates a graph to help "reason" about locations if vector search fails
def build_knowledge_graph(docs):
    print("Building In-Memory Knowledge Graph...")
    G = nx.Graph()
    
    # Simple heuristic to extract Event -> Venue relationships from the text
    # In a real production system, you'd use an LLM to extract these triples.
    # Here we use basic keyword matching from your specific MD structure.
    for doc in docs:
        content = doc.page_content
        # Heuristic: Look for lines like "| Event Name | Venue |"
        lines = content.split('\n')
        for line in lines:
            if "|" in line and "Event" not in line and "---" not in line:
                parts = [p.strip() for p in line.split('|') if p.strip()]
                if len(parts) >= 2:
                    event_name = parts[0]
                    venue_name = parts[1]
                    # Add nodes and edge
                    G.add_node(event_name, type="Event")
                    G.add_node(venue_name, type="Venue")
                    G.add_edge(event_name, venue_name, relation="hosted_at")
    
    print(f"Graph Built: {G.number_of_nodes()} nodes, {G.number_of_edges()} connections.")
    return G

def search_graph(G, query):
    """
    Searches the graph for entities mentioned in the query and returns neighbors.
    Example: Query "Where is Robowars?" -> Finds "Robowars" node -> Returns "OAT".
    """
    context_strings = []
    query_lower = query.lower()
    
    for node in G.nodes():
        if node.lower() in query_lower:
            # Found a matching entity! Get its neighbors (Venues/Events)
            neighbors = list(G.neighbors(node))
            if neighbors:
                context_strings.append(f"GRAPH FACT: {node} is connected to {', '.join(neighbors)}.")
    
    return "\n".join(context_strings)

# --- 3. RETRIEVER SETUP ---

def create_advanced_retriever(file_path, embedding_model):
    print(f"Loading document from {file_path}...")
    loader = TextLoader(file_path)
    docs = loader.load()

    # 1. Build Graph (Side-channel memory)
    kg = build_knowledge_graph(docs)

    # 2. Markdown Splitter (Parent Chunks)
    headers_to_split_on = [
        ("#", "Header 1"),
        ("##", "Header 2"),
        ("###", "Header 3"),
    ]
    md_splitter = MarkdownHeaderTextSplitter(headers_to_split_on=headers_to_split_on)
    parent_docs = md_splitter.split_text(docs[0].page_content)

    # 3. Child Splitter
    child_splitter = RecursiveCharacterTextSplitter(chunk_size=400, chunk_overlap=50)

    # 4. Vectorstore
    vectorstore = Chroma(
        collection_name="shaastra_hybrid", 
        embedding_function=embedding_model,
        persist_directory="./chroma_db_final" 
    )

    store = InMemoryStore()

    print("Creating Hierarchical Retriever...")
    base_retriever = ParentDocumentRetriever(
        vectorstore=vectorstore,
        docstore=store,
        child_splitter=child_splitter,
        parent_splitter=RecursiveCharacterTextSplitter(chunk_size=2000),
        search_kwargs={"k": 15} 
    )

    print(f"Indexing {len(parent_docs)} parent sections...")
    base_retriever.add_documents(parent_docs, ids=None)
    
    return base_retriever, kg

# --- 4. RERANKER SETUP ---

def add_reranker(base_retriever):
    print(f"Loading Reranker Model ({RERANKER_MODEL_ID})...")
    model = HuggingFaceCrossEncoder(model_name=RERANKER_MODEL_ID)
    compressor = CrossEncoderReranker(model=model, top_n=5)
    
    return ContextualCompressionRetriever(
        base_compressor=compressor, 
        base_retriever=base_retriever
    )

# --- 5. LLM LOADING ---

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