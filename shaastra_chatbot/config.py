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
AUDIO_DIR = BASE_DIR / "audio_temp" 

# Ensure audio dir exists
os.makedirs(AUDIO_DIR, exist_ok=True)

# --- MODEL CONFIGURATION ---
LLM_MODEL_ID = "unsloth/llama-3.2-3b-instruct-bnb-4bit"
EMBEDDING_MODEL_ID = "BAAI/bge-base-en-v1.5"
RERANKER_MODEL_ID = "cross-encoder/ms-marco-MiniLM-L-6-v2"

# --- AUDIO MODELS ---
# STT: Whisper (Use 'base.en' to save RAM, or 'small' if you have space)
WHISPER_MODEL_ID = "base.en" 

# TTS: AI4Bharat Indic Parler
PARLER_MODEL_ID = "ai4bharat/indic-parler-tts"

# --- TTS VOICE DESCRIPTION ---
# This controls the accent and language. 
# We default to English with an Indian accent, but you can change 'English' to 'Hindi' or 'Tamil' dynamically.
TTS_DESCRIPTION = "A female speaker delivers a slightly expressive and animated speech in English with a clear Indian accent."

# --- PROMPTS ---
RAG_SYSTEM_PROMPT = """<|begin_of_text|><|start_header_id|>system<|end_header_id|>
You are an intelligent assistant for Shaastra 2025.
Use the Context and Chat History to answer.

RULES:
1. If the user asks "Where is it?" or "When is it?", look at the CHAT HISTORY.
2. If the Context contains conflicting info, prioritize "Event Details" over "Schedule".
3. Think step-by-step before answering.
4. **IMPORTANT: ALWAYS reply in the SAME LANGUAGE as the User's Question.**

<|eot_id|><|start_header_id|>user<|end_header_id|>
CHAT HISTORY:
{chat_history}

CONTEXT:
{context}

QUESTION: {question}<|eot_id|><|start_header_id|>assistant<|end_header_id|>
Thinking Process:
1. Identify language.
2. Identify entity.
3. Formulate answer.

Answer:
"""

ROUTER_PROMPT = """<|begin_of_text|><|start_header_id|>system<|end_header_id|>
You are the primary router for the Shaastra 2025 Techfest Chatbot.

INSTRUCTIONS:
1. Return 'rag' if the question is about Shaastra events, logistics, venues, or follow-ups.
2. Return 'general' ONLY if the question is code, math, or generic greeting.

RETURN ONLY ONE WORD: 'rag' or 'general'.
<|eot_id|><|start_header_id|>user<|end_header_id|>
CHAT HISTORY:
{history}

CURRENT QUESTION: {question}<|eot_id|><|start_header_id|>assistant<|end_header_id|>"""

SUMMARY_PROMPT = """<|begin_of_text|><|start_header_id|>system<|end_header_id|>
Summarize the conversation, keeping details like names, dates, and venues.
<|eot_id|><|start_header_id|>user<|end_header_id|>
SUMMARY: {summary}
NEW LINES: {new_lines}
New Summary:<|eot_id|><|start_header_id|>assistant<|end_header_id|>"""