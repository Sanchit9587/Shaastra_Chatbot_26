import requests
import json

URL = "http://localhost:8000/chat"

def chat_loop():
    print("--- Shaastra API Client ---")
    print("Type 'exit' to quit.\n")
    
    user_id = "test_user_1"
    
    while True:
        query = input("You: ")
        if query.lower() == 'exit':
            break
            
        payload = {
            "user_id": user_id,
            "text": query
        }
        
        try:
            response = requests.post(URL, json=payload)
            response.raise_for_status()
            data = response.json()
            print(f"Bot: {data['response']}\n")
        except Exception as e:
            print(f"Error: {e}")

if __name__ == "__main__":
    chat_loop()