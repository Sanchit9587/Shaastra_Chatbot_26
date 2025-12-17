
# 🐯 Shaastra 2025 Multimodal RAG Chatbot



An intelligent, **local-first** multimodal AI companion for **Shaastra 2025**, the annual technical festival of IIT Madras. This bot allows users to query event details, schedules, and venues using **natural voice** or text, powered by a completely local RAG (Retrieval Augmented Generation) pipeline.



## 🚀 Key Features

*   **🗣️ Full Voice Interaction:** Speak to the bot and hear it speak back using Indian-accented Neural TTS.
*   **🧠 Agentic RAG:** Uses a graph-based state machine (LangGraph) to Route, Rewrite, and Retrieve queries intelligently.
*   **📚 Hybrid Search:** Combines Semantic Vector Search (ChromaDB) with a Knowledge Graph (NetworkX) for precise fact retrieval.
*   **💾 Context Aware:** Remembers conversation history and summarizes it dynamically to maintain context.
*   **⚡ Hardware Optimized:** Designed specifically to run on consumer hardware (6GB VRAM) by balancing GPU/CPU offloading.

---

## 🏗️ System Architecture

The system operates on a Client-Server model to separate heavy model inference from the UI.



    User((User)) -->|Voice/Text| UI[Streamlit Frontend]
    UI -->|HTTP Request| API[FastAPI Backend]
    
    subgraph "Local Inference Engine"
        API --> AudioEngine
        API --> AgentBrain
        
        subgraph "Audio Processing"
            AudioEngine -->|Input| STT[Faster-Whisper (CPU)]
            AudioEngine -->|Output| TTS[Parler-TTS (GPU)]
        end
        
        subgraph "Reasoning Core (LangGraph)"
            AgentBrain --> Router{Router}
            Router -->|General| LLM[Llama 3.2 3B]
            Router -->|Event Info| Retriever
            
            Retriever -->|Hybrid Search| DB[(ChromaDB + Graph)]
            Retriever --> Generator[RAG Generator]
        end
    end
    
    Generator -->|Text Response| UI
    TTS -->|Audio Response| UI


## 🛠️ Tech Stack

| Component | Technology | Model / Library |
| :--- | :--- | :--- |
| **LLM** | Quantized Llama | `unsloth/llama-3.2-3b-instruct-bnb-4bit` |
| **Orchestrator** | State Machine | `LangGraph` |
| **Embedding** | Vector Model | `BAAI/bge-base-en-v1.5` |
| **Database** | Vector Store | `ChromaDB` (Persistent) |
| **STT** | Speech-to-Text | `Faster-Whisper` (Medium.en) |
| **TTS** | Text-to-Speech | `AI4Bharat Indic-Parler` |
| **Backend** | API Framework | `FastAPI` + `Uvicorn` |
| **Frontend** | Interface | `Streamlit` |

---

## ⚙️ Installation

### Prerequisites
*   **OS:** Windows (via WSL2) or Linux.
*   **Python:** 3.10 or higher.
*   **GPU:** NVIDIA GPU with CUDA installed (Minimum 6GB VRAM recommended).
*   **System Tools:** `ffmpeg` (required for audio processing).

### 1. Clone the Repository
```bash
git clone https://github.com/your-username/shaastra-chatbot.git
cd shaastra-chatbot
```

### 2. Install Dependencies
```bash
# Recommended: Create a virtual environment first
python -m venv venv
source venv/bin/activate

# Install packages
pip install -r requirements.txt
```

### 3. Setup Knowledge Base
Ensure your event data is in `rag_context.md`. If this is your first run, the system will automatically parse this file and build the Vector Database.

If you update the markdown file, reset the DB:
```bash
rm -rf chroma_db_prod
```

---

## 🏃‍♂️ How to Run

You need to run the **Backend** and the **Frontend** in separate terminals.

### Terminal 1: Backend API
This loads the AI models into memory. Wait until you see "API is listening".
```bash
python api.py
```

### Terminal 2: Frontend UI
This launches the web interface.
```bash
streamlit run ui.py
```
👉 **Open your browser at:** `http://localhost:8501`

---

## 📂 Project Structure

```text
shaastra_chatbot/
├── api.py               # FastAPI entry point
├── client.py            # CLI Client for testing without UI
├── ui.py                # Streamlit Web Interface
├── audio_engine.py      # Handles Whisper (STT) and Parler (TTS)
├── graph.py             # Defines the LangGraph workflow
├── nodes.py             # Logic for RAG nodes (Retrieve, Generate, etc.)
├── retrieval.py         # ChromaDB and Knowledge Graph setup
├── config.py            # Global paths and settings
├── requirements.txt     # Python dependencies
└── rag_context.md       # Source data for the chatbot
```

---

## ⚠️ Hardware Optimization Notes

This project is tuned for **NVIDIA RTX 3050 (6GB VRAM)**. To prevent Out-Of-Memory (OOM) errors:

1.  **STT Offloading:** `Faster-Whisper` is forced to run on **CPU** (`float32` or `int8`) to save VRAM.
2.  **Quantization:** The LLM is loaded in **4-bit** mode using `bitsandbytes`.
3.  **VRAM Management:** The TTS engine explicitly clears the CUDA cache before generation.
4.  **Latency:** Expect a 3-5 second delay for voice responses as the GPU switches context between the LLM and the TTS model.

---

## 🤝 Contributing

1.  Fork the repo.
2.  Update `rag_context.md` with new events.
3.  Submit a Pull Request.

---

**Built for Shaastra 2025** 🐯
