from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.documents import Document
import config
import components 
from retrieval import search_graph

# --- UTILS ---
def clean_llm_output(text):
    """Removes any CoT or header artifacts."""
    if "Answer:" in text: return text.split("Answer:")[-1].strip()
    if "assistant" in text: return text.split("assistant")[-1].strip()
    return text.strip()

# --- NODES ---

def summarize_conversation(state):
    history = state.get("chat_history", [])
    current_summary = state.get("summary", "")
    
    # Only summarize if history is getting long to save time
    if len(history) > 6:
        print("---SUMMARIZING MEMORY---")
        older_messages = "\n".join(history[:-2]) 
        prompt = PromptTemplate(template=config.SUMMARY_PROMPT, input_variables=["summary", "new_lines"])
        chain = prompt | components.llm | StrOutputParser()
        new_summary = chain.invoke({"summary": current_summary, "new_lines": older_messages})
        return {"summary": new_summary, "chat_history": history[-2:]}
    return {"summary": current_summary, "chat_history": history}

def route_query(state):
    question = state["question"]
    print(f"---ROUTING: '{question}'---")
    
    # 1. Fast Keyword Check (Latency optimization)
    q_lower = question.lower()
    if any(x in q_lower for x in ["hi", "hello", "thanks", "bye", "help"]):
        if len(question.split()) < 5: # Short greeting
            return {"intent": "general"}
            
    # 2. LLM Router (Accuracy)
    prompt = PromptTemplate(template=config.ROUTER_PROMPT, input_variables=["question"])
    chain = prompt | components.llm | StrOutputParser()
    intent = chain.invoke({"question": question}).strip().lower()
    
    # Safety fallback
    if "rag" in intent: return {"intent": "rag"}
    return {"intent": "general"}

def rewrite_query(state):
    question = state["question"]
    history = state.get("chat_history", [])
    
    # If no history, just use the raw question but expand it slightly
    if not history:
        # Simple heuristic expansion
        if "food" in question.lower() and "court" not in question.lower():
            return {"standalone_question": question + " food court dining canteen"}
        return {"standalone_question": question}

    print("---REWRITING---")
    context_str = f"Summary: {state.get('summary', '')}\nRecent: {history}"
    prompt = PromptTemplate(template=config.REWRITE_PROMPT, input_variables=["history", "question"])
    chain = prompt | components.llm | StrOutputParser()
    
    standalone = clean_llm_output(chain.invoke({"history": context_str, "question": question}))
    print(f"📝 Rewritten: {standalone}")
    return {"standalone_question": standalone}

def retrieve(state):
    print("---RETRIEVING---")
    question = state.get("standalone_question", state["question"])
    
    # 1. Vector Search
    documents = components.retriever.invoke(question)
    
    # 2. Graph Search (Boosts Keywords)
    graph_context = search_graph(components.knowledge_graph, question)
    if graph_context:
        print(f"🕸️ Graph Hit: {len(graph_context)} chars")
        graph_doc = Document(page_content=f"**VERIFIED KNOWLEDGE GRAPH FACTS**:\n{graph_context}", metadata={"source": "graph"})
        documents.insert(0, graph_doc) # Put graph data first!
        
    return {"documents": documents}

def generate_rag(state):
    print("---GENERATING (RAG)---")
    documents = state.get("documents", [])
    question = state["question"]
    
    # Fail-fast if no docs found (Prevents Hallucination)
    if not documents:
        return {"generation": "I'm sorry, I couldn't find any information about that in the Shaastra documents. Please check the website or help desk."}

    # Format context
    doc_context = "\n\n".join([f"{d.page_content}" for d in documents])
    
    prompt = PromptTemplate(
        template=config.RAG_SYSTEM_PROMPT, 
        input_variables=["chat_history", "context", "question"]
    )
    
    chain = prompt | components.llm | StrOutputParser()
    raw_gen = chain.invoke({
        "chat_history": "\n".join(state["chat_history"]), 
        "context": doc_context, 
        "question": question
    })
    
    return {"generation": clean_llm_output(raw_gen)}

def generate_general(state):
    print("---GENERATING (GENERAL)---")
    prompt = PromptTemplate(template=config.GENERAL_SYSTEM_PROMPT, input_variables=["question"])
    chain = prompt | components.llm | StrOutputParser()
    raw_gen = chain.invoke({"question": state["question"]})
    return {"generation": clean_llm_output(raw_gen)}