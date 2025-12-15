from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.documents import Document
import config
import components 
from retrieval import search_graph

# --- MEMORY MANAGEMENT NODES ---

def summarize_conversation(state):
    """
    Summarizes the conversation if it gets too long.
    """
    history = state.get("chat_history", [])
    current_summary = state.get("summary", "")
    
    # Trigger summarization only if history > 4 turns to save compute
    if len(history) > 4:
        print("---SUMMARIZING MEMORY---")
        # Take the oldest messages to summarize
        older_messages = "\n".join(history[:-2]) 
        
        prompt = PromptTemplate(
            template=config.SUMMARY_PROMPT,
            input_variables=["summary", "new_lines"]
        )
        chain = prompt | components.llm | StrOutputParser()
        new_summary = chain.invoke({"summary": current_summary, "new_lines": older_messages})
        
        # Keep only the 2 most recent messages in active history
        # Store the rest in the summary
        return {"summary": new_summary, "chat_history": history[-2:]}
    
    return {"summary": current_summary, "chat_history": history}


# --- EXISTING NODES (Updated to use Summary) ---

def route_query(state):
    print("---ROUTING---")
    question = state["question"]
    # Simple keyword check for speed
    if any(x in question.lower() for x in ["hi", "hello", "code", "math"]):
        return {"intent": "general"}
    return {"intent": "rag"} # Default to RAG for safety

def rewrite_query(state):
    print("---REWRITING---")
    question = state["question"]
    # Use summary + recent history for context
    context_str = f"Summary: {state.get('summary', '')}\nRecent: {state['chat_history']}"
    
    standalone = components.rewriter_chain.invoke({"history": context_str, "question": question})
    if "Standalone Question:" in standalone:
        standalone = standalone.split("Standalone Question:")[-1].strip()
    return {"standalone_question": standalone}

def retrieve(state):
    print("---RETRIEVING---")
    question = state.get("standalone_question", state["question"])
    documents = components.retriever.invoke(question)
    
    graph_context = search_graph(components.knowledge_graph, question)
    if graph_context:
        print("🕸️ Graph Context Injected")
        graph_doc = Document(page_content=f"**GRAPH FACT**:\n{graph_context}", metadata={"Header 1": "Graph"})
        documents.insert(0, graph_doc)
        
    return {"documents": documents}

def generate_rag(state):
    print("---GENERATING (RAG)---")
    question = state["question"]
    documents = state["documents"]
    summary = state.get("summary", "No previous context.")
    history = "\n".join(state["chat_history"])

    if not documents: return {"generation": "I couldn't find info on that."}

    doc_context = "\n\n".join([f"[Source: {d.metadata}]\n{d.page_content}" for d in documents])
    
    prompt = PromptTemplate(
        template=config.RAG_SYSTEM_PROMPT, 
        input_variables=["summary", "chat_history", "context", "question"]
    )
    
    chain = prompt | components.llm | StrOutputParser()
    gen = chain.invoke({
        "summary": summary,
        "chat_history": history, 
        "context": doc_context, 
        "question": question
    })
    
    # Strip CoT if present in output
    if "Answer:" in gen: gen = gen.split("Answer:")[-1].strip()
    
    return {"generation": gen}

def generate_general(state):
    print("---GENERATING (GENERAL)---")
    prompt = PromptTemplate(template="<|begin_of_text|><|start_header_id|>system<|end_header_id|>Helpful AI.<|eot_id|><|start_header_id|>user<|end_header_id|>{q}<|eot_id|><|start_header_id|>assistant<|end_header_id|>", input_variables=["q"])
    chain = prompt | components.llm | StrOutputParser()
    return {"generation": chain.invoke({"q": state["question"]})}