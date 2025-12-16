# main.py
# Import config FIRST to apply SQLite Fix
import config 
from graph import build_graph

# Global Memory
MEMORY = []

if __name__ == "__main__":
    print("🚀 Initializing Shaastra 2025 Bot...")
    app = build_graph()
    print("\n💡 Bot is Online! (Type 'exit' to stop)")

    while True:
        try:
            user_query = input("\nUser: ")
            if not user_query.strip(): continue
            if user_query.lower() == 'exit': break
            
            inputs = {"question": user_query, "chat_history": MEMORY}
            result = app.invoke(inputs)
            
            ans = result['generation']
            
            # --- Chain of Thought Cleanup ---
            # The model will output "Thinking Process: ... Answer: ..."
            # We want to show just the answer to the user, or keep it if debugging.
            if "Answer:" in ans:
                ans = ans.split("Answer:")[-1].strip()
            
            print(f"\n🤖 Bot: {ans}")
            
            MEMORY.append(f"User: {user_query}")
            MEMORY.append(f"AI: {ans}")
            
        except KeyboardInterrupt:
            break
        except Exception as e:
            print(f"Error: {e}")