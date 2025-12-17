import requests
import base64
import os

# --- CONFIGURATION ---
API_URL = "http://localhost:8000/chat/audio"
INPUT_FILE = "test_audio.wav"
OUTPUT_FILE = "bot_reply.wav"
USER_ID = "test_user_1"

def test_audio_file():
    # 1. Check if input file exists
    if not os.path.exists(INPUT_FILE):
        print(f"❌ Error: '{INPUT_FILE}' not found.")
        print("Please record a short audio file, name it 'test_audio.wav', and place it in this folder.")
        return

    print(f"🎤 Sending '{INPUT_FILE}' to Shaastra API...")
    
    try:
        # 2. Send Request
        with open(INPUT_FILE, "rb") as f:
            files = {"file": f}
            data = {"user_id": USER_ID}
            response = requests.post(API_URL, files=files, data=data)
        
        # 3. Handle Response
        if response.status_code == 200:
            res = response.json()
            
            print("\n" + "="*40)
            print(f"🗣️  Transcription: {res.get('user_query')}")
            print(f"🤖 AI Response:   {res.get('text_response')}")
            print("="*40)
            
            # 4. Save Audio
            if res.get('audio_base64'):
                audio_bytes = base64.b64decode(res['audio_base64'])
                with open(OUTPUT_FILE, "wb") as f:
                    f.write(audio_bytes)
                print(f"\n✅ Audio response saved to: {OUTPUT_FILE}")
            else:
                print("\n⚠️ No audio data in response.")
                
        else:
            print(f"\n❌ Server Error: {response.status_code}")
            print(response.text)

    except Exception as e:
        print(f"\n❌ Connection Error: {e}")
        print("Ensure 'python api.py' is running.")

if __name__ == "__main__":
    test_audio_file()