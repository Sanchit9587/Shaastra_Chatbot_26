import requests
import base64

API_URL = "http://localhost:8000/chat/audio"
# Make sure you have a file named 'test_audio.wav' in the same folder!
AUDIO_FILE = "test_audio.wav" 

def test_audio():
    print("🎤 Sending Audio...")
    
    with open(AUDIO_FILE, "rb") as f:
        files = {"file": f}
        data = {"user_id": "tester_1"}
        response = requests.post(API_URL, files=files, data=data)
    
    if response.status_code == 200:
        res = response.json()
        print(f"🗣️  You said: {res['user_query']}")
        print(f"🤖 Bot said: {res['text_response']}")
        
        # Save output audio
        audio_data = base64.b64decode(res['audio_base64'])
        with open("bot_reply.mp3", "wb") as f:
            f.write(audio_data)
        print("🔊 Audio saved to 'bot_reply.mp3'")
    else:
        print("Error:", response.text)

if __name__ == "__main__":
    test_audio()
    