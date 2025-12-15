import config
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from contextlib import asynccontextmanager # <--- NEW IMPORT
# Import components only when needed or keep global if preferred
from components import context, llm, retriever, knowledge_graph, rewriter_chain 
from graph import build_graph

# --- LIFESPAN MANAGER (Fixes Warning) ---
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup logic
    print("🚀 API Starting... Loading Models...")
    # You can force model load here if lazy loading is preferred
    print("✅ Models Ready. API is listening.")
    yield
    # Shutdown logic (optional: clear VRAM)
    print("🛑 API Shutting down.")

# Initialize API with lifespan
app = FastAPI(title="Shaastra 2025 RAG API", lifespan=lifespan)

# Initialize Graph
bot_app = build_graph()

# Global Memory
SESSION_MEMORY = {
    "default": {
        "history": [],
        "summary": ""
    }
}

class ChatRequest(BaseModel):
    user_id: str = "default"
    text: str

class ChatResponse(BaseModel):
    response: str

@app.post("/chat", response_model=ChatResponse)
async def chat_endpoint(request: ChatRequest):
    user_id = request.user_id
    query = request.text
    
    # Init session if new
    if user_id not in SESSION_MEMORY:
        SESSION_MEMORY[user_id] = {"history": [], "summary": ""}
        
    user_state = SESSION_MEMORY[user_id]
    
    try:
        inputs = {
            "question": query,
            "chat_history": user_state["history"],
            "summary": user_state["summary"]
        }
        
        result = bot_app.invoke(inputs)
        ans = result['generation']
        
        # Clean specific CoT artifacts if any remain
        if "Answer:" in ans:
            ans = ans.split("Answer:")[-1].strip()

        # Update Memory
        new_history = result.get('chat_history', user_state["history"])
        # Append new turn
        new_history.append(f"User: {query}")
        new_history.append(f"AI: {ans}")
        
        # Keep sliding window in session
        if len(new_history) > 6:
            new_history = new_history[-6:]

        SESSION_MEMORY[user_id]["history"] = new_history
        SESSION_MEMORY[user_id]["summary"] = result.get("summary", "")
        
        return ChatResponse(response=ans)

    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)