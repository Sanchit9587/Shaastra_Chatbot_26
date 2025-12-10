# --- FIX FOR CHROMA DB ---
try:
    __import__('pysqlite3')
    import sys
    sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')
except ImportError:
    pass
# -------------------------

import sys
from typing import TypedDict, List, Literal, Any 
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langgraph.graph import END, StateGraph
from langchain_core.messages import HumanMessage, AIMessage

from advanced_rag import create_advanced_retriever, add_reranker, search_graph, load_llm, LLM_CONFIG, CONTEXT_FILE
from langchain_huggingface import HuggingFaceEmbeddings
import torch

# --- 1. SETUP STATE ---
class GraphState(TypedDict):
    question: str
    generation: str
    documents: List[Any]
    chat_history: List[str] # Memory stores string summary of history
    intent: str 

# --- 2. INITIALIZE MODELS ---
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Initializing Embeddings on {device}...")
embedding_model = HuggingFaceEmbeddings(model_name='BAAI/bge-base-en-v1.5', model_kwargs={'device': device})

# Initialize Retriever AND Knowledge Graph
base_retriever, knowledge_graph = create_advanced_retriever(CONTEXT_FILE, embedding_model)
retriever = add_reranker(base_retriever)

config = LLM_CONFIG["Unsloth 3B"]
llm = load_llm(config["model_id"])

# Global memory list (simple implementation for local run)
MEMORY = []

# --- 3. HELPER FUNCTIONS ---
def format_history(history):
    return "\n".join(history[-4:]) # Keep last 4 turns

# --- 4. DEFINE NODES ---

def route_query(state):
    print("---ROUTING QUERY---")
    question = state["question"]
    
    # Smarter Router Prompt
    router_prompt = PromptTemplate(
        template="""<|begin_of_text|><|start_header_id|>system<|end_header_id|>
Classify the user question.
1. 'rag': Questions about Shaastra events, dates, venues, food, workshops, rules, or hackathons.
2. 'general': Greetings (hi, hello), coding questions (write python code), math, or general knowledge.
3. 'followup': If the user asks "Where is it?" or "Tell me more" referring to previous chat.

Return ONLY one word: 'rag', 'general', or 'followup'.<|eot_id|><|start_header_id|>user<|end_header_id|>
Question: {question}<|eot_id|><|start_header_id|>assistant<|end_header_id|>""",
        input_variables=["question"]
    )
    
    chain = router_prompt | llm | StrOutputParser()
    decision = chain.invoke({"question": question}).strip().lower()
    
    # Cleaning decision
    if "rag" in decision: decision = "rag"
    elif "followup" in decision: decision = "rag" # Followups usually need context too
    else: decision = "general"
        
    print(f"---DECISION: {decision.upper()}---")
    return {"intent": decision}

def retrieve(state):
    print("---RETRIEVING (HYBRID: VECTOR + GRAPH)---")
    question = state["question"]
    
    # 1. Vector Search + Reranking
    documents = retriever.invoke(question)
    
    # 2. Knowledge Graph Search
    graph_context = search_graph(knowledge_graph, question)
    
    # 3. Inject Graph Context into Metadata of first doc (hack to pass it along)
    if graph_context and documents:
        documents[0].page_content = f"__GRAPH_KNOWLEDGE__:\n{graph_context}\n\n" + documents[0].page_content
        print(f"🕸️ Graph found connections: {graph_context}")
    
    return {"documents": documents}

def generate_rag(state):
    print("---GENERATING (CoT + MEMORY)---")
    question = state["question"]
    documents = state["documents"]
    history_str = format_history(state["chat_history"])
    
    # Format Context
    formatted_context = ""
    for doc in documents:
        headers = ", ".join([f"{k}: {v}" for k, v in doc.metadata.items() if k.startswith("Header")])
        formatted_context += f"\n[Source: {headers}]\n{doc.page_content}\n"
    
    prompt = PromptTemplate(
        template=config["prompt_template"],
        input_variables=["chat_history", "context", "question"]
    )
    
    chain = prompt | llm | StrOutputParser()
    generation = chain.invoke({
        "chat_history": history_str, 
        "context": formatted_context, 
        "question": question
    })
    
    return {"generation": generation}

def generate_general(state):
    print("---GENERATING (GENERAL)---")
    question = state["question"]
    
    prompt = PromptTemplate(
        template="""<|begin_of_text|><|start_header_id|>system<|end_header_id|>
You are a helpful AI assistant. Answer politely and concisely.
<|eot_id|><|start_header_id|>user<|end_header_id|>
{question}<|eot_id|><|start_header_id|>assistant<|end_header_id|>""",
        input_variables=["question"]
    )
    
    chain = prompt | llm | StrOutputParser()
    generation = chain.invoke({"question": question})
    return {"generation": generation}

# --- 5. BUILD GRAPH ---

workflow = StateGraph(GraphState)

workflow.add_node("router", route_query)
workflow.add_node("retrieve", retrieve)
workflow.add_node("generate_rag", generate_rag)
workflow.add_node("generate_general", generate_general)

workflow.set_entry_point("router")

def route_decision(state):
    return state["intent"]

workflow.add_conditional_edges("router", route_decision, {"rag": "retrieve", "general": "generate_general"})
workflow.add_edge("retrieve", "generate_rag")
workflow.add_edge("generate_rag", END)
workflow.add_edge("generate_general", END)

app = workflow.compile()

# --- 6. EXECUTION LOOP ---

if __name__ == "__main__":
    print("\n💡 Shaastra 2025 AI is Online. (Memory Enabled). Type 'exit' to stop.")
    
    while True:
        try:
            user_query = input("\nUser: ")
            if user_query.lower() == 'exit': break
            
            # Run Graph
            inputs = {
                "question": user_query, 
                "chat_history": MEMORY
            }
            final_res = app.invoke(inputs)
            
            # Clean Output
            raw_ans = final_res['generation']
            if "Answer:" in raw_ans:
                clean_ans = raw_ans.split("Answer:")[-1].strip()
            else:
                clean_ans = raw_ans

            print(f"\n🤖 Bot: {clean_ans}")
            
            # Update Memory
            MEMORY.append(f"User: {user_query}")
            MEMORY.append(f"AI: {clean_ans}")
            
        except KeyboardInterrupt:
            print("\nExiting...")
            break
        except Exception as e:
            print(f"Error: {e}")
            import traceback
            traceback.print_exc()