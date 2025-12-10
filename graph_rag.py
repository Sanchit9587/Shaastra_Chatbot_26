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

# Import setup
from advanced_rag import create_advanced_retriever, add_reranker, search_graph, load_llm, LLM_CONFIG, CONTEXT_FILE
from langchain_huggingface import HuggingFaceEmbeddings
import torch

# --- 1. SETUP STATE ---
class GraphState(TypedDict):
    question: str
    generation: str
    documents: List[Any]
    chat_history: List[str] 
    intent: str 

# --- 2. INITIALIZE MODELS ---
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Initializing Embeddings on {device}...")
embedding_model = HuggingFaceEmbeddings(model_name='BAAI/bge-base-en-v1.5', model_kwargs={'device': device})

# Initialize Hybrid Engines
base_retriever, knowledge_graph = create_advanced_retriever(CONTEXT_FILE, embedding_model)
retriever = add_reranker(base_retriever)

config = LLM_CONFIG["Unsloth 3B"]
llm = load_llm(config["model_id"])

# Global memory
MEMORY = []

# --- 3. HELPER FUNCTIONS ---
def format_history(history):
    # Get last 3 exchanges to keep context window small but relevant
    return "\n".join(history[-6:]) 

# --- 4. DEFINE NODES ---

def route_query(state):
    print("---ROUTING QUERY---")
    question = state["question"]
    history = format_history(state["chat_history"])
    
    # --- IMPROVED ROUTER PROMPT ---
    # We explicitly tell it to assume 'rag' for vague questions if they fit the festival context
    router_prompt = PromptTemplate(
        template="""<|begin_of_text|><|start_header_id|>system<|end_header_id|>
You are the primary router for the Shaastra 2025 Techfest Chatbot.
Your job is to decide if a user's question requires looking up the Shaastra database or if it is unrelated.

CONTEXTUAL AWARENESS:
The user is attending the fest. If they ask "Where is food?" or "Where is the toilet?", they mean AT SHAASTRA.

INSTRUCTIONS:
1. Return 'rag' if the question is about:
   - Events, schedules, venues, dates, workshops, competitions.
   - Logistics (food, accommodation, parking, wifi, bathrooms).
   - Vague questions like "Where is it?" (Assume it refers to the event).
   - Follow-up questions based on chat history.

2. Return 'general' ONLY if the question is:
   - A pure coding task (e.g., "Write a merge sort in Python").
   - A math problem (e.g., "What is 2+2?").
   - A general greeting (e.g., "Hi", "Good morning") with no other text.
   - Completely unrelated to the event (e.g., "Who is the President of USA?").

RETURN ONLY ONE WORD: 'rag' or 'general'.

<|eot_id|><|start_header_id|>user<|end_header_id|>
CHAT HISTORY:
{history}

CURRENT QUESTION: {question}<|eot_id|><|start_header_id|>assistant<|end_header_id|>""",
        input_variables=["history", "question"]
    )
    
    chain = router_prompt | llm | StrOutputParser()
    decision = chain.invoke({"history": history, "question": question}).strip().lower()
    
    # Hard rules to prevent router failure on simple things
    if "rag" in decision: decision = "rag"
    else: decision = "general"
        
    print(f"---DECISION: {decision.upper()}---")
    return {"intent": decision}

def retrieve(state):
    print("---RETRIEVING (HYBRID: VECTOR + GRAPH)---")
    question = state["question"]
    
    # 1. Vector Search + Reranking (Standard)
    documents = retriever.invoke(question)
    
    # 2. Knowledge Graph Search (Enhanced)
    # Use graph to find connections if Vector search feels weak or if specific entities are named
    graph_context = search_graph(knowledge_graph, question)
    
    if graph_context:
        print(f"🕸️ Graph Knowledge Injected: Found connections for entities in query.")
        # Create a fake document to inject graph knowledge at the top
        from langchain_core.documents import Document
        graph_doc = Document(
            page_content=f"**KNOWLEDGE GRAPH CONNECTIONS**:\n{graph_context}",
            metadata={"Header 1": "Knowledge Graph", "Header 2": "Direct Relationships"}
        )
        # Insert graph knowledge at position 0 so LLM sees it first
        documents.insert(0, graph_doc)
    
    return {"documents": documents}

def generate_rag(state):
    print("---GENERATING (CoT + MEMORY)---")
    question = state["question"]
    documents = state["documents"]
    history_str = format_history(state["chat_history"])
    
    # Format Context nicely with headers
    formatted_context = ""
    for doc in documents:
        # Extract headers if available, otherwise default
        headers = ", ".join([f"{v}" for k, v in doc.metadata.items() if k.startswith("Header")])
        if not headers: headers = "General Context"
        
        formatted_context += f"\n[Section: {headers}]\n{doc.page_content}\n"
    
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
You are a helpful AI assistant. Answer the user's question directly, politely, and concisely.
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
    print("\n💡 Shaastra 2025 AI is Online. (Memory & Intelligent Routing Enabled). Type 'exit' to stop.")
    
    while True:
        try:
            user_query = input("\nUser: ")
            if not user_query.strip(): continue # Skip empty enters
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