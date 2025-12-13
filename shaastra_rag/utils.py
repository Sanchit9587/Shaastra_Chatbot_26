import torch

def check_gpu_status():
    print("\n--- SYSTEM CHECK ---")
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        print(f"✅ GPU Detected: {gpu_name}")
        return 'cuda'
    print("❌ No GPU detected. Running on CPU.")
    return 'cpu'

def format_history(history):
    # Keep last 6 turns to manage context window
    return "\n".join(history[-6:]) 

def print_retrieved_chunks(documents):
    print("\n" + "="*40)
    print(f"🔍 DEBUG: Retrieved {len(documents)} Context Chunks")
    print("="*40)
    for i, doc in enumerate(documents):
        # Clean up newlines for display
        content_preview = doc.page_content.replace('\n', ' ')[:200] 
        meta = doc.metadata if hasattr(doc, 'metadata') else {}
        print(f"📄 Chunk {i+1} Metadata: {meta}")
        print(f"📝 Content Preview: {content_preview}...")
        print("-" * 20)
    print("="*40 + "\n")