# hybrid_search.py
import numpy as np
from rank_bm25 import BM25Okapi
from thefuzz import fuzz
from sentence_transformers import SentenceTransformer, util

# --- 1. PREPARE YOUR DATA AND MODELS ---

# Let's use a sample from your rag_context.md as our document chunks
corpus = [
    "Event: Spotlight Lecture by Dr. Sanjeev Das | Date: 2025-01-03 | Time: 11:00-12:00 | Venue: CLT",
    "Event: Fireside Chat with Dr. Sujatha Ramdorai | Date: 2025-01-03 | Time: 15:00-16:00 | Venue: CLT",
    "Workshop: Intro to AI and ML | Date: 2025-01-03 | Time: 09:00-17:00 | Venue: CRC 103 | Fee: 499/-",
    "Event: Robowars | Dates: 2025-01-03, 2025-01-04 | Time: 19:00-22:00 | Venue: OAT | Sponsor: IDFC & BAJAJ",
    "Rule: Possession and consumption of alcohol is strictly prohibited. The participant status will be null and void."
]

# Load your models
embedding_model = SentenceTransformer('all-MiniLM-L6-v2')

# --- 2. SETUP THE RETRIEVERS ---

# a) Dense Retriever (Vector Search)
print("Creating embeddings for dense search...")
corpus_embeddings = embedding_model.encode(corpus, convert_to_tensor=True)

# b) Sparse Retriever (BM25 Keyword Search)
print("Tokenizing corpus for sparse search...")
tokenized_corpus = [doc.lower().split(" ") for doc in corpus]
bm25 = BM25Okapi(tokenized_corpus)

# --- 3. DEFINE THE SEARCH AND RAG FUNCTIONS ---

def dense_search(query, top_k=3):
    query_embedding = embedding_model.encode(query, convert_to_tensor=True)
    cos_scores = util.cos_sim(query_embedding, corpus_embeddings)[0]
    top_results = np.argpartition(-cos_scores, range(top_k))[0:top_k]
    return [(corpus[idx], cos_scores[idx].item()) for idx in top_results]

def sparse_search(query, top_k=3):
    tokenized_query = query.lower().split(" ")
    doc_scores = bm25.get_scores(tokenized_query)
    top_results_indices = np.argpartition(-doc_scores, range(top_k))[0:top_k]
    return [(corpus[idx], doc_scores[idx]) for idx in top_results_indices]

def fuzzy_match_and_rerank(query, chunks):
    """Simple fuzzy matching to boost scores of chunks with close keywords."""
    reranked = []
    for chunk, score in chunks:
        # Get a ratio of how similar the query is to the chunk text
        fuzz_ratio = fuzz.partial_ratio(query.lower(), chunk.lower()) / 100.0
        # Boost original score by the fuzzy match ratio (this is a simple heuristic)
        new_score = score + (score * fuzz_ratio * 0.2) # Add a 20% boost based on fuzzy match
        reranked.append((chunk, new_score))
    return sorted(reranked, key=lambda x: x[1], reverse=True)

def mock_llm_call(query, context_chunk):
    """This is a mock function. In reality, you would call the Gemini API here."""
    print("\n--- LLM INPUT ---")
    print(f"QUERY: {query}")
    print(f"CONTEXT: {context_chunk}")
    print("-----------------\n")
    # A fake response for demonstration
    response = f"Based on the provided context, the event with Dr. Das is a Spotlight Lecture at 11:00 in the CLT."
    return response

# --- 4. EXECUTE THE HYBRID SEARCH RAG PIPELINE ---

if __name__ == "__main__":
    # Query with a typo to test fuzzy search
    user_query = "lecture with dr sanjev das"
    print(f"Executing search for query: '{user_query}'\n")

    # Step A: Retrieve from both systems
    dense_results = dense_search(user_query)
    sparse_results = sparse_search(user_query)

    print("--- Dense Search Results (Semantic) ---")
    for doc, score in dense_results:
        print(f"Score: {score:.4f} | Doc: {doc}")

    print("\n--- Sparse Search Results (Keyword) ---")
    for doc, score in sparse_results:
        print(f"Score: {score:.4f} | Doc: {doc}")

    # Step B: Combine and de-duplicate results
    combined_results = {doc: score for doc, score in dense_results}
    for doc, score in sparse_results:
        # Use the higher score if a doc is found by both
        if doc not in combined_results or score > combined_results[doc]:
            combined_results[doc] = score
    
    combined_results = list(combined_results.items())

    # Step C: Re-rank using Fuzzy Matching to handle the typo "sanjev"
    reranked_results = fuzzy_match_and_rerank(user_query, combined_results)

    print("\n--- Final Re-ranked Results (Hybrid + Fuzzy) ---")
    for doc, score in reranked_results:
        print(f"Score: {score:.4f} | Doc: {doc}")
        
    # Step D: Select the top result and pass ONLY that to the LLM
    top_chunk = reranked_results[0][0]
    final_answer = mock_llm_call(user_query, top_chunk)
    
    print(f"Final Answer from LLM: {final_answer}")