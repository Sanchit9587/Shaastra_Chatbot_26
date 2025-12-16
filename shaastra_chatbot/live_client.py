import sounddevice as sd
import scipy.io.wavfile as wav
import requests
import base64
import os
import numpy as np
import time

# --- CONFIGURATION ---
API_URL = "http://localhost:8000/chat/audio"
USER_ID = "voice_user"
INPUT_FILENAME = "mic_input.wav"
RESPONSE_FILENAME = "bot_reply.wav"
SAMPLE_RATE = 16000  # Whisper works best with 16kHz
DURATION = 7         # How long to record (in seconds)

def record_audio(duration=DURATION):
    print("\n" + "="*40)
    print(f"🎤 GET READY! Recording for {duration} seconds...")
    print("="*40)
    time.sleep(1) # Give a split second to prepare
    
    print("🔴 SPEAK NOW...")
    # Record audio (Mono channel)
    recording = sd.rec(int(duration * SAMPLE_RATE), samplerate=SAMPLE_RATE, channels=1, dtype=np.int16)
    sd.wait()  # Wait until recording is finished
    print("⏹️  Recording stopped.")
    
    # Save to file
    wav.write(INPUT_FILENAME, SAMPLE_RATE, recording)
    print(f"✅ Audio saved to {INPUT_FILENAME}")

def send_and_play():
    if not os.path.exists(INPUT_FILENAME):
        print("❌ No recording found.")
        return

    print("🚀 Sending audio to Shaastra Bot...")
    
    try:
        with open(INPUT_FILENAME, "rb") as f:
            files = {"file": f}
            data = {"user_id": USER_ID}
            response = requests.post(API_URL, files=files, data=data)
        
        if response.status_code == 200:
            res = response.json()
            
            # 1. Print what the bot heard and answered
            print("\n" + "-"*40)
            print(f"🗣️  You said: {res['user_query']}")
            print(f"🤖 Bot Text: {res['text_response']}")
            print("-" * 40)
            
            # 2. Save the audio response
            if res['audio_base64']:
                print("🔊 Playing Audio Response...")
                audio_data = base64.b64decode(res['audio_base64'])
                with open(RESPONSE_FILENAME, "wb") as f:
                    f.write(audio_data)
                
                # 3. Play audio using ffplay (Part of ffmpeg)
                # -nodisp: No window, -autoexit: Close when done, -hide_banner: Less log text
                os.system(f"ffplay -nodisp -autoexit -hide_banner {RESPONSE_FILENAME} >/dev/null 2>&1")
            else:
                print("⚠️ No audio received from bot.")
        else:
            print(f"❌ Server Error: {response.text}")
            
    except Exception as e:
        print(f"❌ Connection Error: {e}")
        print("Is `python api.py` running in another terminal?")

if __name__ == "__main__":
    while True:
        choice = input("\nPress [Enter] to Record, or type 'q' to quit: ")
        if choice.lower() == 'q':
            break
        
        record_audio()
        send_and_play()