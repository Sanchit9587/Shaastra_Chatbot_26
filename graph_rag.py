# --- FIX FOR CHROMA DB ---
try:
    __import__('pysqlite3')
    import sys
    sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')
except ImportError:
    pass
# -------------------------

import sys
# Included Any to fix NameError
from typing import TypedDict, List, Literal, Any 
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langgraph.graph import END, StateGraph

from advanced_rag import create_hierarchical_retriever, load_llm, LLM_CONFIG, CONTEXT_FILE
from langchain_huggingface import HuggingFaceEmbeddings
import torch

# --- 1. SETUP STATE ---
class GraphState(TypedDict):
    question: str
    generation: str
    documents: List[Any]
    intent: str 

# --- 2. INITIALIZE MODELS ---
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Initializing Embeddings on {device}...")
# Using BGE-Base for better retrieval accuracy than MiniLM
embedding_model = HuggingFaceEmbeddings(model_name='BAAI/bge-base-en-v1.5', model_kwargs={'device': device})

# Initialize Retriever
retriever = create_hierarchical_retriever(CONTEXT_FILE, embedding_model)

# Initialize LLM
config = LLM_CONFIG["Unsloth 3B"]
llm = load_llm(config["model_id"])

# --- 3. HELPER: CHUNK VISUALIZER ---
def print_retrieved_chunks(documents):
    print("\n" + "="*40)
    print(f"🔍 DEBUG: Retrieved {len(documents)} Context Chunks")
    print("="*40)
    for i, doc in enumerate(documents):
        # Clean up newlines for display
        content_preview = doc.page_content.replace('\n', ' ')[:200] 
        print(f"📄 Chunk {i+1} Metadata: {doc.metadata}")
        print(f"📝 Content Preview: {content_preview}...")
        print("-" * 20)
    print("="*40 + "\n")

# --- 4. DEFINE NODES ---

def route_query(state):
    """
    Decides if the query is about Shaastra (needs RAG) or General (needs LLM only).
    """
    print("---ROUTING QUERY---")
    question = state["question"]
    
    router_prompt = PromptTemplate(
        template="""<|begin_of_text|><|start_header_id|>system<|end_header_id|>
You are a router. Classify the user question.
If it is about 'Shaastra', 'events', 'workshops', 'dates', 'venues', 'moot court', 'hackathons', 'schedule' or 'rules', return 'rag'.
If it is a general greeting, math problem, or code question, return 'general'.
Return ONLY the word 'rag' or 'general'.<|eot_id|><|start_header_id|>user<|end_header_id|>
Question: {question}<|eot_id|><|start_header_id|>assistant<|end_header_id|>""",
        input_variables=["question"]
    )
    
    chain = router_prompt | llm | StrOutputParser()
    decision = chain.invoke({"question": question}).strip().lower()
    
    # Fallback safety
    if "rag" in decision:
        decision = "rag"
    else:
        decision = "general"
        
    print(f"---DECISION: {decision.upper()}---")
    return {"intent": decision}

def retrieve(state):
    print("---RETRIEVING---")
    question = state["question"]
    # This retrieves the PARENT docs (Full sections) based on child matches
    documents = retriever.invoke(question)
    
    # DEBUG VISUALIZATION
    print_retrieved_chunks(documents)
    
    return {"documents": documents}

def generate_rag(state):
    print("---GENERATING (RAG)---")
    question = state["question"]
    documents = state["documents"]
    
    context = "\n\n".join([doc.page_content for doc in documents])
    
    prompt = PromptTemplate(
        template=config["prompt_template"],
        input_variables=["system_instruction", "context", "question"]
    )
    
    # Inject specific system instruction for RAG
    system_inst = "You must answer using ONLY the context provided. If the context doesn't contain the answer, say you don't know."
    
    chain = prompt | llm | StrOutputParser()
    generation = chain.invoke({"system_instruction": system_inst, "context": context, "question": question})
    return {"generation": generation}

def generate_general(state):
    print("---GENERATING (GENERAL)---")
    question = state["question"]
    
    prompt = PromptTemplate(
        template=config["prompt_template"],
        input_variables=["system_instruction", "context", "question"]
    )
    
    # Inject general chat system instruction
    system_inst = "You are a helpful AI assistant. Answer the user's question directly."
    
    chain = prompt | llm | StrOutputParser()
    # Pass empty context
    generation = chain.invoke({"system_instruction": system_inst, "context": "No specific context required.", "question": question})
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

workflow.add_conditional_edges(
    "router",
    route_decision,
    {
        "rag": "retrieve",
        "general": "generate_general"
    }
)

workflow.add_edge("retrieve", "generate_rag")
workflow.add_edge("generate_rag", END)
workflow.add_edge("generate_general", END)

app = workflow.compile()

# --- 6. EXECUTION ---

if __name__ == "__main__":
    print("\n💡 Shaastra Bot is ready! (Type 'exit' to stop)")
    while True:
        try:
            user_query = input("\nUser: ")
            if user_query.lower() == 'exit':
                break
            
            inputs = {"question": user_query}
            final_res = app.invoke(inputs)
            
            print(f"\n🤖 Bot: {final_res['generation']}")
            
        except KeyboardInterrupt:
            print("\nExiting...")
            break
        except Exception as e:
            print(f"Error: {e}")
            import traceback
            traceback.print_exc()