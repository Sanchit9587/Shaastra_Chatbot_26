import os
import sys
from pathlib import Path

# --- CRITICAL FIX FOR CHROMA DB ---
try:
    __import__('pysqlite3')
    sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')
except ImportError:
    pass
# ----------------------------------

# Paths
BASE_DIR = Path(__file__).parent.resolve()
CONTEXT_FILE = BASE_DIR / "rag_context.md"
CHROMA_DB_DIR = BASE_DIR / "chroma_db" / "final"

# --- MODEL CONFIGURATION ---
# WARNING: 11B Vision requires ~8GB VRAM. Your RTX 3050 has 6GB.
# I recommend sticking to 3B for speed and stability.
# LLM_MODEL_ID = "unsloth/Llama-3.2-11B-Vision-Instruct-bnb-4bit" 
LLM_MODEL_ID = "unsloth/llama-3.2-3b-instruct-bnb-4bit"

EMBEDDING_MODEL_ID = "BAAI/bge-base-en-v1.5"
RERANKER_MODEL_ID = "cross-encoder/ms-marco-MiniLM-L-6-v2"

# --- PROMPTS ---

# 1. System Prompt (Now includes Summary)
RAG_SYSTEM_PROMPT = """<|begin_of_text|><|start_header_id|>system<|end_header_id|>
You are an intelligent assistant for Shaastra 2025.

MEMORY CONTEXT:
{summary}

RECENT CHAT HISTORY:
{chat_history}

RETRIEVED KNOWLEDGE:
{context}

RULES:
1. Use the MEMORY and HISTORY to understand context.
2. Answer the user's QUESTION using the RETRIEVED KNOWLEDGE.
3. If specific info isn't in the knowledge, say "I don't know".
<|eot_id|><|start_header_id|>user<|end_header_id|>
{question}<|eot_id|><|start_header_id|>assistant<|end_header_id|>"""

# 2. Router Prompt
ROUTER_PROMPT = """<|begin_of_text|><|start_header_id|>system<|end_header_id|>
Classify the user question.
Context: Shaastra 2025 Techfest.

INTENTS:
- 'rag': Events, schedule, food, venues, rules, or specific fest details.
- 'general': Greetings, coding, math, or generic chatter.

RETURN ONLY: 'rag' or 'general'.
<|eot_id|><|start_header_id|>user<|end_header_id|>
{question}<|eot_id|><|start_header_id|>assistant<|end_header_id|>"""

# 3. Summarization Prompt (From PDF Strategy)
SUMMARY_PROMPT = """<|begin_of_text|><|start_header_id|>system<|end_header_id|>
Progressively summarize the lines of conversation provided, adding to the previous summary returning a new summary.

CURRENT SUMMARY:
{summary}

NEW LINES:
{new_lines}

New Summary:<|eot_id|><|start_header_id|>assistant<|end_header_id|>"""