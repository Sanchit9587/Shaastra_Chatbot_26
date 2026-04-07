import config
import os
import csv
import shutil
from datetime import datetime
from pathlib import Path
os.environ["TF_USE_LEGACY_KERAS"] = "1"

from fastapi import FastAPI, HTTPException, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from contextlib import asynccontextmanager
from graph import build_graph
from audio_engine import AudioEngine

# --- GLOBAL COMPONENTS ---
audio_engine = None
bot_app = None
SESSION_MEMORY = {}

# CSV Config (Extra Feature)
LOG_FILE = Path("shaastra_chat_logs.csv")
CSV_COLUMNS = ["Timestamp", "User_ID", "Input_Mode", "User_Query", "AI_Response"]

# --- EXTRA FEATURE: LOGGING UTILITY ---
def init_csv():
    """Initializes CSV with headers if not present."""
    if not LOG_FILE.exists():
        with open(LOG_FILE, mode='w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(CSV_COLUMNS)

def log_interaction(user_id: str, query: str, response: str, mode: str):
    """Appends interaction to CSV."""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    try:
        with open(LOG_FILE, mode='a', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow([timestamp, user_id, mode, query, response])
    except Exception as e:
        print(f"Logging error: {e}")

# --- LIFESPAN MANAGER ---
@asynccontextmanager
async def lifespan(app: FastAPI):
    global bot_app, audio_engine
    print("🚀 API Starting... Loading AI Models...")
    
    # Run CSV init without changing model loading flow
    init_csv()
    
    import components 
    bot_app = build_graph()
    audio_engine = AudioEngine()
    
    print("✅ All Models Ready. API is listening.")
    yield
    print("🛑 API Shutting down.")

app = FastAPI(title="Shaastra 2025 Multimodal API", lifespan=lifespan)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- DATA MODELS ---
class ChatRequest(BaseModel):
    user_id: str = "default"
    text: str

class ChatResponse(BaseModel):
    text_response: str
    audio_base64: str = None

# --- CORE LOGIC ---
def process_query(user_id, query):
    if user_id not in SESSION_MEMORY:
        SESSION_MEMORY[user_id] = {"history": [], "summary": ""}
    
    user_state = SESSION_MEMORY[user_id]
    
    inputs = {
        "question": query,
        "chat_history": user_state["history"],
        "summary": user_state["summary"]
    }
    
    result = bot_app.invoke(inputs)
    ans = result['generation']
    
    if "Answer:" in ans: ans = ans.split("Answer:")[-1].strip()
    
    new_history = result.get('chat_history', [])
    new_history.append(f"User: {query}")
    new_history.append(f"AI: {ans}")
    
    SESSION_MEMORY[user_id]["history"] = new_history
    SESSION_MEMORY[user_id]["summary"] = result.get("summary", user_state["summary"])
    
    return ans

# --- ENDPOINTS ---

@app.post("/chat/text", response_model=ChatResponse)
async def chat_text_endpoint(request: ChatRequest):
    """Text In -> Text Out"""
    try:
        response_text = process_query(request.user_id, request.text)
        
        # New Feature: Log Text Query
        log_interaction(request.user_id, request.text, response_text, "text")
        
        return ChatResponse(text_response=response_text)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/chat/audio")
async def chat_audio_endpoint(
    user_id: str = Form(...),
    file: UploadFile = File(...)
):
    """Audio In -> Audio + Text Out"""
    try:
        temp_filename = f"{user_id}_input.wav"
        temp_path = config.AUDIO_DIR / temp_filename
        
        with open(temp_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
            
        print(f"🎤 Transcribing audio for {user_id}...")
        transcribed_text = audio_engine.speech_to_text(str(temp_path))
        
        if not transcribed_text:
            return {"text_response": "I couldn't hear anything.", "audio_base64": None}

        response_text = process_query(user_id, transcribed_text)
        
        # New Feature: Log Audio Query
        log_interaction(user_id, transcribed_text, response_text, "audio")
        
        audio_b64 = None
        if config.ENABLE_TTS and getattr(audio_engine, "tts_enabled", False):
            output_audio_path = config.AUDIO_DIR / f"{user_id}_response.wav"
            tts_text = response_text[:500] 
            audio_path = audio_engine.text_to_speech(tts_text, str(output_audio_path))
            if audio_path:
                audio_b64 = audio_engine.audio_file_to_base64(output_audio_path)

        return {
            "user_query": transcribed_text,
            "text_response": response_text,
            "audio_base64": audio_b64
        }

    except Exception as e:
        print(f"Error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=6006)