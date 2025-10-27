# advanced_rag.py
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, pipeline
import textwrap
import sys

# LangChain Imports
from langchain.prompts import PromptTemplate
from langchain.schema.runnable import RunnablePassthrough
from langchain.schema.output_parser import StrOutputParser
from langchain.text_splitter import MarkdownHeaderTextSplitter
from langchain_community.document_loaders import TextLoader
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.llms.huggingface_pipeline import HuggingFacePipeline
from langchain.retrievers import BM25Retriever, EnsembleRetriever
from langchain.retrievers.document_compressors import CrossEncoderReranker
from langchain.retrievers import ContextualCompressionRetriever
from langchain_community.cross_encoders import HuggingFaceCrossEncoder

# --- 1. LLM AND EMBEDDING MODEL CONFIGURATION ---

# Define the LLMs you want to use. The first one is the default.
LLM_CONFIG = {
    "Unsloth 3B (Cached)": {
        "model_id": "unsloth/llama-3.2-3b-instruct-bnb-4bit",
        "prompt_template": """<|begin_of_text|><|start_header_id|>system<|end_header_id|>

You are a helpful and precise assistant for the Shaastra 2025 tech festival.
You must answer the user's question using ONLY the context provided.
If there are multiple dates or venues, list them all. If the answer is not in the context, say "I do not have enough information to answer this." Do not make up information.<|eot_id|><|start_header_id|>user<|end_header_id|>

CONTEXT:
---
{context}
---

QUESTION: {question}<|eot_id|><|start_header_id|>assistant<|end_header_id|>
"""
    },
    "Zephyr 7B Beta": {
        "model_id": "HuggingFaceH4/zephyr-7b-beta",
        "prompt_template": """<|system|>
You are a helpful and precise assistant for the Shaastra 2025 tech festival.
You must answer the user's question using ONLY the context provided.
If there are multiple dates or venues, list them all. If the answer is not in the context, say "I do not have enough information to answer this." Do not make up information.</s>
<|user|>
CONTEXT:
---
{context}
---

QUESTION: {question}</s>
<|assistant|>
"""
    }
}

EMBEDDING_MODEL_ID = 'all-MiniLM-L6-v2'
RERANKER_MODEL_ID = 'BAAI/bge-reranker-base'
CONTEXT_FILE = "rag_context.md"

# --- 2. ADVANCED DOCUMENT CHUNKING ---

def load_and_chunk_document(file_path):
    """Loads a Markdown document and splits it into chunks based on headers."""
    print("Loading and chunking document...")
    loader = TextLoader(file_path)
    docs = loader.load()

    headers_to_split_on = [
        ("#", "Header 1"),
        ("##", "Header 2"),
        ("###", "Header 3"),
    ]

    text_splitter = MarkdownHeaderTextSplitter(headers_to_split_on=headers_to_split_on)
    chunks = text_splitter.split_text(docs[0].page_content)
    
    # Optional: You can print a sample chunk to see the result
    # print("\n--- Sample Chunk ---")
    # print(chunks[5].page_content)
    # print(chunks[5].metadata)
    # print("--------------------\n")
    
    print(f"Document split into {len(chunks)} logical chunks.")
    return chunks

# --- 3. ADVANCED RETRIEVER SETUP (Hybrid Search + Reranker) ---

def create_advanced_retriever(chunks, embedding_model):
    """Creates a sophisticated retriever with hybrid search and a reranker."""
    print("Creating advanced retriever...")

    # a. Vector Store (for semantic search)
    vectorstore = Chroma.from_documents(chunks, embedding_model)
    chroma_retriever = vectorstore.as_retriever(search_kwargs={"k": 10})

    # b. BM25 Retriever (for keyword search)
    bm25_retriever = BM25Retriever.from_documents(chunks)
    bm25_retriever.k = 10

    # c. Ensemble Retriever (combines the two)
    ensemble_retriever = EnsembleRetriever(
        retrievers=[bm25_retriever, chroma_retriever],
        weights=[0.5, 0.5] # Give equal weight to keyword and semantic search
    )
    
    # d. Reranker (takes top results and re-ranks for relevance)
    cross_encoder = HuggingFaceCrossEncoder(model_name=RERANKER_MODEL_ID)
    compressor = CrossEncoderReranker(model=cross_encoder, top_n=3) # Return top 3 most relevant
    
    compression_retriever = ContextualCompressionRetriever(
        base_compressor=compressor, base_retriever=ensemble_retriever
    )
    
    print("Retriever setup complete.")
    return compression_retriever

# --- 4. LLM LOADING ---

def load_llm(model_id):
    """Loads the specified Hugging Face model with 4-bit quantization."""
    print(f"Loading LLM: {model_id}...")
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
    hf_pipeline = pipeline("text-generation", model=model, tokenizer=tokenizer, max_new_tokens=300, return_full_text=False)
    # Wrap the pipeline for LangChain compatibility
    return HuggingFacePipeline(pipeline=hf_pipeline)

# --- 5. MAIN EXECUTION ---

def main():
    # --- User Model Selection ---
    print("Please choose a local LLM to load:")
    model_names = list(LLM_CONFIG.keys())
    for i, name in enumerate(model_names):
        print(f"{i + 1}: {name}")
    
    try:
        choice = int(input(f"Enter your choice (1-{len(model_names)}): ")) - 1
        if choice < 0 or choice >= len(model_names):
            raise ValueError
        chosen_model_name = model_names[choice]
    except (ValueError, IndexError):
        print("Invalid choice. Loading default model.")
        chosen_model_name = model_names[0]

    config = LLM_CONFIG[chosen_model_name]
    
    # --- Pipeline Setup ---
    llm = load_llm(config["model_id"])
    docs = load_and_chunk_document(CONTEXT_FILE)
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    embedding_model = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_ID, model_kwargs={'device': device})
    
    retriever = create_advanced_retriever(docs, embedding_model)
    
    prompt = PromptTemplate(template=config["prompt_template"], input_variables=["context", "question"])
    
    # --- LangChain Expression Language (LCEL) RAG Chain ---
    rag_chain = (
        {"context": retriever, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )

    print(f"\n--- Shaastra 2025 Chatbot Ready (Model: {chosen_model_name}) ---")
    while True:
        user_query = input("\nPlease ask a question (or type 'exit' to quit): ")
        if user_query.lower() == 'exit':
            break

        print("\nFinding relevant information and generating answer...")
        
        # --- (Optional) For Debugging: See what the retriever finds ---
        # retrieved_docs = retriever.get_relevant_documents(user_query)
        # print("\n--- Retrieved Context ---")
        # for i, doc in enumerate(retrieved_docs):
        #     print(f"--- Document {i+1} ---\n{doc.page_content}\n")
        # print("------------------------\n")
        
        # --- Invoke the chain to get the final answer ---
        try:
            answer = rag_chain.invoke(user_query)
            print("\n--- Answer ---")
            print("\n".join(textwrap.wrap(answer, width=90)))
            print("--------------\n")
        except Exception as e:
            print(f"An error occurred during generation: {e}")

if __name__ == "__main__":
    main()