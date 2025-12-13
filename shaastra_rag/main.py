# main.py
import config
from models import load_llm, load_embedding_model
from retriever.hierarchical import create_hierarchical_retriever
from retriever.reranker import add_reranker
from chains.query_rewriter import create_query_rewriter
from chains.grader import create_grader
from graph.workflow import build_app, RAGContext

# Global Memory
MEMORY = []

def initialize_system():
    print("🚀 Initializing Models & Retriever...")
    
    # 1. Load Models
    llm = load_llm(config.LLM_MODEL_ID)
    embed_model = load_embedding_model(config.EMBEDDING_MODEL_ID)
    
    # 2. Setup Retrievers
    base_retriever, kg = create_hierarchical_retriever(config.CONTEXT_FILE, embed_model)
    final_retriever = add_reranker(base_retriever)
    
    # 3. Setup Chains
    rewriter = create_query_rewriter(llm)
    grader = create_grader(llm)
    
    # 4. Context Object
    context = RAGContext(llm, final_retriever, kg, rewriter, grader)
    
    return build_app(context)

if __name__ == "__main__":
    app = initialize_system()
    print("\n💡 Shaastra 2025 AI is Online. (Type 'exit' to stop)")

    while True:
        try:
            user_query = input("\nUser: ")
            if not user_query.strip(): continue
            if user_query.lower() == 'exit': break
            
            inputs = {"question": user_query, "chat_history": MEMORY}
            result = app.invoke(inputs)
            
            ans = result['generation']
            # Cleanup CoT traces
            if "Answer:" in ans: ans = ans.split("Answer:")[-1].strip()
            
            print(f"\n🤖 Bot: {ans}")
            
            MEMORY.append(f"User: {user_query}")
            MEMORY.append(f"AI: {ans}")
            
        except KeyboardInterrupt:
            print("\nExiting...")
            break
        except Exception as e:
            print(f"Error: {e}")