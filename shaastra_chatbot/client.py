import os
import time
import json
import base64
import requests
import numpy as np
import sounddevice as sd
import scipy.io.wavfile as wav

# --- CONFIGURATION ---
# 'localhost' works from Windows to WSL automatically
BASE_URL = "http://localhost:8000" 
USER_ID = "shaastra_visitor"
SAMPLE_RATE = 16000
RECORD_SECONDS = 7 
TEMP_INPUT = "temp_input.wav"
TEMP_OUTPUT = "temp_output.wav"

class Colors:
    # Colors might not work in standard Windows CMD, but work in PowerShell/VSCode
    HEADER = '\033[95m'
    BLUE = '\033[94m'
    GREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'

def print_bot(text):
    print(f"\n🤖 Shaastra AI: {text}")

def print_user(text):
    print(f"👤 You: {text}")

# --- AUDIO HANDLERS ---

def record_audio(filename=TEMP_INPUT, duration=RECORD_SECONDS):
    print(f"\n🔴 Recording for {duration} seconds... Speak Now!")
    
    # Record mono audio at 16kHz
    recording = sd.rec(int(duration * SAMPLE_RATE), samplerate=SAMPLE_RATE, channels=1, dtype=np.int16)
    sd.wait()
    
    print("⏹️  Recording Finished.")
    wav.write(filename, SAMPLE_RATE, recording)

def play_audio(filename):
    print("🔊 Playing Response...")
    try:
        # Read WAV file
        samplerate, data = wav.read(filename)
        # Play using sounddevice (Cross-platform)
        sd.play(data, samplerate)
        sd.wait()
    except Exception as e:
        print(f"❌ Error playing audio: {e}")

# --- NETWORK HANDLERS ---

def send_text(query):
    url = f"{BASE_URL}/chat/text"
    try:
        response = requests.post(url, json={"user_id": USER_ID, "text": query})
        response.raise_for_status()
        data = response.json()
        print_bot(data['text_response'])
    except Exception as e:
        print(f"❌ Error: {e}")

def send_audio():
    url = f"{BASE_URL}/chat/audio"
    if not os.path.exists(TEMP_INPUT):
        print("❌ Audio file not found.")
        return

    print("🚀 Sending Audio to API...")
    
    try:
        with open(TEMP_INPUT, "rb") as f:
            files = {"file": f}
            data = {"user_id": USER_ID}
            response = requests.post(url, files=files, data=data)
        
        response.raise_for_status()
        res = response.json()
        
        # 1. Show Transcription
        print_user(f"(Transcribed): {res.get('user_query', '???')}")
        
        # 2. Show Text Response
        print_bot(res.get('text_response', '...'))
        
        # 3. Play Audio Response
        if res.get('audio_base64'):
            audio_bytes = base64.b64decode(res['audio_base64'])
            with open(TEMP_OUTPUT, "wb") as f:
                f.write(audio_bytes)
            play_audio(TEMP_OUTPUT)
        else:
            print("⚠️ No audio response received.")

    except Exception as e:
        print(f"❌ Connection Error: {e}")

# --- MAIN UI ---

def main():
    print("\n=======================================")
    print("   Shaastra 2025 Multimodal Client     ")
    print("=======================================")
    
    while True:
        print("\nSelect Mode:")
        print("1. 📝 Text Chat")
        print("2. 🎙️  Voice Chat")
        print("3. 🚪 Exit")
        
        choice = input("\nOption (1/2/3): ").strip()
        
        if choice == '1':
            query = input("\nType your question: ")
            if query:
                send_text(query)
        
        elif choice == '2':
            input("\nPress [Enter] to start recording...")
            record_audio()
            send_audio()
            
        elif choice == '3':
            print("Goodbye! 👋")
            break
        else:
            print("Invalid option.")

        # Cleanup
        if os.path.exists(TEMP_INPUT): os.remove(TEMP_INPUT)
        if os.path.exists(TEMP_OUTPUT): os.remove(TEMP_OUTPUT)

if __name__ == "__main__":
    main()