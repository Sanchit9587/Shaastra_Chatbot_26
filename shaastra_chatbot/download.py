import os
from huggingface_hub import snapshot_download

def download_all_models():
    print("🚀 Starting Model Downloads...")

    # 1. Indic Parler TTS (Gated)
    print("\n⬇️  Downloading AI4Bharat Indic Parler TTS...")
    try:
        snapshot_download(
            repo_id="ai4bharat/indic-parler-tts",
            local_dir_use_symlinks=False  # Safer for WSL
        )
        print("✅ Indic Parler TTS Downloaded.")
    except Exception as e:
        print(f"❌ Failed to download Indic Parler TTS. Did you accept the license and login? Error: {e}")

    # 2. Faster Whisper (Medium)
    print("\n⬇️  Downloading Faster Whisper (Medium)...")
    try:
        snapshot_download(repo_id="systran/faster-whisper-medium.en")
        print("✅ Faster Whisper Downloaded.")
    except Exception as e:
        print(f"❌ Failed to download Whisper: {e}")

    # 3. Reranker (MiniLM)
    print("\n⬇️  Downloading Reranker...")
    try:
        snapshot_download(repo_id="cross-encoder/ms-marco-MiniLM-L-6-v2")
        print("✅ Reranker Downloaded.")
    except Exception as e:
        print(f"❌ Failed to download Reranker: {e}")

if __name__ == "__main__":
    download_all_models()