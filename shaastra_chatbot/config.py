import os
import sys
from pathlib import Path

# =============================================================================
# 🔧 SYSTEM COMPATIBILITY FIXES
# =============================================================================
# Critical fix for ChromaDB on systems with older SQLite versions (like default Linux)
try:
    __import__('pysqlite3')
    sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')
except ImportError:
    pass

# =============================================================================
# 📂 DIRECTORY PATHS
# =============================================================================
BASE_DIR = Path(__file__).parent.resolve()

# The Markdown file containing all Shaastra data (Source of Truth)
CONTEXT_FILE = BASE_DIR / "rag_context.md"

# Persistent Vector Database Path (Set to 'prod' for the improved logic)
CHROMA_DB_DIR = BASE_DIR / "chroma_db_prod"

# Temporary storage for Audio I/O
AUDIO_DIR = BASE_DIR / "audio_temp" 
os.makedirs(AUDIO_DIR, exist_ok=True)

# =============================================================================
# 🧠 AI MODEL CONFIGURATION
# =============================================================================

# LLM: 4-bit Quantized Llama 3.2 (Optimized for 6GB VRAM)
LLM_MODEL_ID = "unsloth/llama-3.2-3b-instruct-bnb-4bit"

# Embeddings: High performance, small size
EMBEDDING_MODEL_ID = "BAAI/bge-base-en-v1.5"

# Reranker: Used to filter bad retrieval results
RERANKER_MODEL_ID = "cross-encoder/ms-marco-MiniLM-L-6-v2"

# =============================================================================
# 🎧 AUDIO MODEL CONFIGURATION (Fixes your AttributeError)
# =============================================================================

# STT: Faster-Whisper (Runs on CPU to save GPU for LLM)
# Options: "tiny.en", "base.en", "small.en", "medium.en"
WHISPER_MODEL_ID = "base.en" 

# TTS: Parler TTS (Runs on GPU)
PARLER_MODEL_ID = "ai4bharat/indic-parler-tts"

# TTS Voice Description: Defines the accent and style
TTS_DESCRIPTION = "A female speaker delivers a slightly expressive and animated speech in English with a clear Indian accent."

# Toggle to enable/disable TTS functionality (set to False to disable Parler TTS and avoid heavy GPU usage)
ENABLE_TTS = False

# =============================================================================
# 📝 PROMPT ENGINEERING (PHASE 2 UPGRADE)
# =============================================================================

# 1. ROUTER PROMPT
# Uses Few-Shot Semantic Routing to distinguish "Chat" from "Search"
ROUTER_PROMPT = """<|begin_of_text|><|start_header_id|>system<|end_header_id|>
You are a semantic router for the Shaastra 2025 Techfest AI.
Classify the user's input into exactly one of these two categories:

1. 'rag': Queries about events, schedule, venues, tickets, food, rules, prizes, specific people (e.g., "Dr. Sanjeev"), competitions, or logistics.
2. 'general': Greetings, compliments, requests for code, math, jokes, personal questions to the bot, or general world knowledge (e.g., "Capital of France").

EXAMPLES:
User: "When is Robowars?" -> rag
User: "Hi, good morning!" -> general
User: "Write a Python script." -> general
User: "Is there coffee nearby?" -> rag
User: "Who is the Prime Minister?" -> general
User: "Tell me about Envisage." -> rag
User: "Where can I sleep?" -> rag

OUTPUT ONLY THE LABEL ('rag' or 'general'). NO EXPLANATION.<|eot_id|><|start_header_id|>user<|end_header_id|>
Query: {question}<|eot_id|><|start_header_id|>assistant<|end_header_id|>"""

# 2. RAG GENERATION PROMPT
# Forces grounded answers and prevents inventing facts.
RAG_SYSTEM_PROMPT = """<|begin_of_text|><|start_header_id|>system<|end_header_id|>
You are ShaastraBot, the official AI guide for IIT Madras Shaastra 2025.
Answer the user's question using ONLY the context provided below.

STRICT RULES:
1. If the answer is not in the context, say: "I'm sorry, I don't have that specific information in my documents."
2. Do NOT invent dates, venues, or prizes.
3. If the user asks about locations (Food, ATM, Hostels), assume they are asking about inside IIT Madras.
4. Keep answers professional, concise, and helpful.
5. Do NOT mention "According to the context". Just answer naturally.

CONTEXT:
{context}

CHAT HISTORY:
{chat_history}

QUESTION: {question}<|eot_id|><|start_header_id|>assistant<|end_header_id|>"""

# 3. GENERAL CHAT PROMPT
# Persona Lock: Prevents the bot from thinking it is a 40-year-old man.
GENERAL_SYSTEM_PROMPT = """<|begin_of_text|><|start_header_id|>system<|end_header_id|>
You are ShaastraBot, a helpful AI assistant for the IIT Madras Techfest.
You are NOT a human. You do NOT have a personal life, age, or gender.

INSTRUCTIONS:
- If the user greets you, welcome them to Shaastra 2025 and ask how you can help.
- If asked general questions (math, code, science), answer concisely.
- If asked about your identity, say: "I am the Shaastra 2025 AI Assistant."
- Refuse to roleplay as a human (e.g., do not describe your physical appearance or dating life).

USER INPUT: {question}<|eot_id|><|start_header_id|>assistant<|end_header_id|>"""

# 4. QUERY REWRITER PROMPT
# Expands queries with synonyms for better retrieval.
REWRITE_PROMPT = """<|begin_of_text|><|start_header_id|>system<|end_header_id|>
Rewrite the user's question to be a specific search query for a vector database.
- Resolve "it", "they", "there" using chat history.
- Add synonyms for retrieval (e.g., "Food" -> "Food Court Canteen Dining").
- If the question is about location, add "Venue".

History: {history}
User Input: {question}

Output ONLY the rewritten query text.<|eot_id|><|start_header_id|>assistant<|end_header_id|>"""

# 5. SUMMARIZATION PROMPT
# Compresses history to save token context.
SUMMARY_PROMPT = """<|begin_of_text|><|start_header_id|>system<|end_header_id|>
Compress the conversation history. Keep specific details like Event Names, Venues, and Dates. Drop greetings and small talk.<|eot_id|><|start_header_id|>user<|end_header_id|>
Current Summary: {summary}
New Lines: {new_lines}
Updated Summary:<|eot_id|><|start_header_id|>assistant<|end_header_id|>"""