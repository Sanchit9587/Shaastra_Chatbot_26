# ollama_rag.py
import ollama
import numpy as np
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer, util
import textwrap
import sys

# --- 1. HYBRID SEARCH RETRIEVER (Optimized) ---

class HybridRetriever:
    def __init__(self, context_file):
        print("Initializing the Hybrid Retriever...")
        self.corpus = self._load_corpus(context_file)
        print(f"Loaded {len(self.corpus)} chunks from the document.")
        
        print("Loading sentence embedding model...")
        # Use a device that's available, preferring GPU if possible for speed
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2', device=device)
        print(f"Embedding model loaded on '{device}'.")

        self.corpus_embeddings = self.embedding_model.encode(self.corpus, convert_to_tensor=True)
        
        print("Tokenizing corpus for BM25...")
        tokenized_corpus = [doc.lower().split(" ") for doc in self.corpus]
        self.bm25 = BM25Okapi(tokenized_corpus)
        print("Retriever initialization complete.")

    def _load_corpus(self, file_path):
        """Loads and chunks the RAG context document."""
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
            # Split by a unique separator and filter out empty strings
            chunks = [chunk.strip() for chunk in content.split("---") if chunk.strip()]
            # Filter for chunks that have a reasonable length
            return [c for c in chunks if len(c) > 30]

    def search(self, query, top_k=5):
        """Performs a hybrid search and returns the top_k most relevant chunks."""
        # Dense search
        query_embedding = self.embedding_model.encode(query, convert_to_tensor=True)
        cos_scores = util.cos_sim(query_embedding, self.corpus_embeddings)[0].cpu().numpy()
        
        # Sparse search
        tokenized_query = query.lower().split(" ")
        bm25_scores = self.bm25.get_scores(tokenized_query)
        # Handle cases where all bm25 scores are zero
        max_bm25 = max(bm25_scores) if max(bm25_scores) > 0 else 1
        bm25_scores_normalized = bm25_scores / max_bm25

        # Hybrid Score (Weighted to slightly prefer semantic search)
        hybrid_scores = (0.6 * cos_scores) + (0.4 * bm25_scores_normalized)
        
        top_indices = np.argsort(hybrid_scores)[-top_k:][::-1]
        return [self.corpus[idx] for idx in top_indices]

# --- 2. MAIN RAG PIPELINE EXECUTION ---

def format_rag_prompt(query, context_chunks):
    """Formats the prompt with retrieved context for Llama 3."""
    context_str = "\n\n".join(context_chunks)
    
    # Llama 3 uses a specific instruction format for best results
    prompt = f"""<|begin_of_text|><|start_header_id|>system<|end_header_id|>

You are a helpful and precise assistant for the Shaastra 2025 tech festival.
You must answer the user's question using ONLY the context provided.
If the answer is not available in the context, you MUST say "I do not have enough information to answer this question." Do not make up information.<|eot_id|><|start_header_id|>user<|end_header_id|>

CONTEXT:
---
{context_str}
---

QUESTION: {query}<|eot_id|><|start_header_id|>assistant<|end_header_id|>
"""
    return prompt

if __name__ == "__main__":
    # === MODEL CONFIGURATION ===
    # This now points to your installed 8B Llama 3.1 model
    OLLAMA_MODEL = 'llama3.1:8b'
    # ===========================

    try:
        # Check if Ollama service is running and if the model is available
        client = ollama.Client()
        available_models = [m['name'] for m in client.list()['models']]
        
        if not available_models:
             print("Error: No models found in Ollama. Please run `ollama pull llama3.1:8b` first.")
             sys.exit()

        if OLLAMA_MODEL not in available_models:
            print(f"Error: The required model '{OLLAMA_MODEL}' was not found in Ollama.")
            print(f"Available models are: {', '.join(available_models)}")
            print(f"Please run `ollama pull {OLLAMA_MODEL}` or edit the OLLAMA_MODEL variable in the script.")
            sys.exit()

    except Exception as e:
        print(f"Error connecting to Ollama: {e}")
        print("Please ensure the Ollama application or `ollama serve` is running in a separate terminal.")
        sys.exit()

    # Import torch here after checks, as it's a heavy import
    import torch
    
    # Setup retriever
    retriever = HybridRetriever("rag_context.md")

    # --- Main Interaction Loop ---
    print(f"\n--- Shaastra 2025 Chatbot Ready (Model: {OLLAMA_MODEL}) ---")
    while True:
        user_query = input("\nPlease ask a question (or type 'exit' to quit): ")
        if user_query.lower() == 'exit':
            break
        
        # 1. Retrieve relevant context
        print("Finding relevant information...")
        retrieved_chunks = retriever.search(user_query, top_k=3)
        
        # 2. Format the prompt
        rag_prompt = format_rag_prompt(user_query, retrieved_chunks)
        
        # 3. Generate the response from Ollama
        print(f"Asking {OLLAMA_MODEL}...")
        try:
            response = client.generate(
                model=OLLAMA_MODEL,
                prompt=rag_prompt,
                stream=False
            )
            final_answer = response['response']
        except Exception as e:
            final_answer = f"An error occurred while generating the answer: {e}"
        
        print("\n--- Answer ---")
        print("\n".join(textwrap.wrap(final_answer, width=80)))
        print("--------------\n")