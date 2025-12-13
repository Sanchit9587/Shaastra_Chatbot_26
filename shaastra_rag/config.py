import os
import sys
from pathlib import Path

# --- CRITICAL FIX FOR CHROMA DB (SQLite FTS5) ---
try:
    __import__('pysqlite3')
    sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')
except ImportError:
    pass
# ------------------------------------------------

# Paths
BASE_DIR = Path(__file__).parent.resolve()
CONTEXT_FILE = BASE_DIR / "rag_context.md"
CHROMA_DB_DIR = BASE_DIR / "chroma_db" / "final"

# Model IDs
LLM_MODEL_ID = "unsloth/llama-3.2-3b-instruct-bnb-4bit"
EMBEDDING_MODEL_ID = "BAAI/bge-base-en-v1.5"
RERANKER_MODEL_ID = "cross-encoder/ms-marco-MiniLM-L-6-v2"

# Prompts
RAG_SYSTEM_PROMPT = """<|begin_of_text|><|start_header_id|>system<|end_header_id|>
You are an intelligent assistant for Shaastra 2025.
Use the Context and Chat History to answer.

RULES:
1. If the user asks "Where is it?" or "When is it?", look at the CHAT HISTORY to resolve the entity.
2. If the Context contains conflicting info, prioritize "Event Details" over "Schedule".
3. Think step-by-step before answering.

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

ROUTER_PROMPT = """<|begin_of_text|><|start_header_id|>system<|end_header_id|>
You are the primary router for the Shaastra 2025 Techfest Chatbot.
Your job is to decide if a user's question requires looking up the Shaastra database.

CONTEXTUAL AWARENESS:
The user is attending the fest. "Where is food?" implies AT SHAASTRA.

INSTRUCTIONS:
1. Return 'rag' if the question is about:
   - Events, schedules, venues, dates, workshops, competitions.
   - Logistics (food, accommodation, parking, wifi, bathrooms).
   - Vague questions like "Where is it?" (Assume it refers to the event).

2. Return 'general' ONLY if the question is:
   - A pure coding task (e.g., "Write a merge sort").
   - A math problem (e.g., "What is 2+2?").
   - A general greeting (e.g., "Hi").

RETURN ONLY ONE WORD: 'rag' or 'general'.
<|eot_id|><|start_header_id|>user<|end_header_id|>
CHAT HISTORY:
{history}

CURRENT QUESTION: {question}<|eot_id|><|start_header_id|>assistant<|end_header_id|>"""