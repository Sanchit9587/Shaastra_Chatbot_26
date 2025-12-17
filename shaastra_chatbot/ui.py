import streamlit as st
import requests
import base64
import time

# --- CONFIGURATION ---
API_URL = "http://localhost:8000"
USER_ID = "ui_test_user"

st.set_page_config(
    page_title="Shaastra 2025 AI",
    page_icon="🤖",
    layout="centered"
)

# --- SESSION STATE ---
if "messages" not in st.session_state:
    st.session_state.messages = []

# --- HELPER FUNCTIONS ---
def autoplay_audio(audio_base64):
    """
    Decodes base64 audio and displays an audio player.
    """
    if audio_base64:
        audio_bytes = base64.b64decode(audio_base64)
        st.audio(audio_bytes, format="audio/wav", start_time=0)

def handle_text_chat():
    """Process text input."""
    user_input = st.chat_input("Type your question about Shaastra...")
    if user_input:
        # 1. Add User Message to UI
        st.session_state.messages.append({"role": "user", "content": user_input})
        
        # 2. Call API
        with st.chat_message("user"):
            st.markdown(user_input)

        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                try:
                    payload = {"user_id": USER_ID, "text": user_input}
                    response = requests.post(f"{API_URL}/chat/text", json=payload)
                    
                    if response.status_code == 200:
                        data = response.json()
                        bot_text = data.get("text_response", "No response.")
                        st.markdown(bot_text)
                        
                        # Save to history
                        st.session_state.messages.append({"role": "assistant", "content": bot_text})
                    else:
                        st.error(f"API Error: {response.status_code}")
                except Exception as e:
                    st.error(f"Connection Error: {e}")

def handle_audio_chat():
    """Process audio input using Streamlit's native audio recorder."""
    audio_value = st.audio_input("Record a voice query")
    
    if audio_value:
        # Check if we already processed this specific audio buffer to prevent re-running on redraw
        # Streamlit re-runs script on interaction, so we need a simple check
        if "last_audio_id" not in st.session_state:
            st.session_state.last_audio_id = None
        
        # Simple hash or check of object identity
        current_audio_id = id(audio_value)
        
        if st.session_state.last_audio_id != current_audio_id:
            st.session_state.last_audio_id = current_audio_id
            
            # 1. Display User Audio
            with st.chat_message("user"):
                st.audio(audio_value)
                st.caption("🎤 Audio Query Sent")
            
            # Add placeholder to history
            st.session_state.messages.append({"role": "user", "content": "🎤 [Audio Query]", "audio_data": audio_value})

            # 2. Call API
            with st.chat_message("assistant"):
                with st.spinner("Listening, Thinking & Speaking..."):
                    try:
                        files = {"file": ("input.wav", audio_value, "audio/wav")}
                        data = {"user_id": USER_ID}
                        
                        # Send to /chat/audio endpoint
                        response = requests.post(f"{API_URL}/chat/audio", files=files, data=data)
                        
                        if response.status_code == 200:
                            res_json = response.json()
                            transcribed = res_json.get("user_query", "")
                            bot_text = res_json.get("text_response", "")
                            audio_b64 = res_json.get("audio_base64", None)
                            
                            # Display text details
                            st.markdown(f"**🗣️ Transcribed:** *{transcribed}*")
                            st.markdown(bot_text)
                            
                            # Play Audio
                            if audio_b64:
                                autoplay_audio(audio_b64)
                            
                            # Update History
                            st.session_state.messages.append({
                                "role": "assistant", 
                                "content": f"🗣️ **User said:** {transcribed}\n\n🤖 **AI:** {bot_text}",
                                "audio_b64": audio_b64
                            })
                            
                        else:
                            st.error(f"Server Error: {response.status_code}")
                            
                    except Exception as e:
                        st.error(f"Connection failed. Is api.py running? {e}")

# --- MAIN UI LAYOUT ---

st.title("🐯 Shaastra 2025 Multimodal Bot")
st.caption(f"Running on Localhost | RTX 3050 Optimized")

# Sidebar for controls
with st.sidebar:
    st.header("Controls")
    if st.button("Clear Chat History"):
        st.session_state.messages = []
        st.rerun()
    
    st.markdown("---")
    st.markdown("**System Status:**")
    try:
        # Quick health check (optional, assumes root endpoint exists or just checks connection)
        requests.get(f"{API_URL}/docs", timeout=1)
        st.success("API Online ✅")
    except:
        st.error("API Offline ❌")
        st.info("Run `python api.py` in terminal.")

# 1. Display Chat History
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])
        # If there was audio in history, show player
        if "audio_data" in msg:
            st.audio(msg["audio_data"])
        if "audio_b64" in msg:
            autoplay_audio(msg.get("audio_b64"))

# 2. Audio Input Section
handle_audio_chat()

# 3. Text Input Section (Bottom)
handle_text_chat()