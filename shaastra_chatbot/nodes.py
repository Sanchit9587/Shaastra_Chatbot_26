from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.documents import Document
import config
import components 
from retrieval import search_graph

# --- MEMORY MANAGEMENT NODES ---

def summarize_conversation(state):
    history = state.get("chat_history", [])
    current_summary = state.get("summary", "")
    if len(history) > 4:
        print("---SUMMARIZING MEMORY---")
        older_messages = "\n".join(history[:-2]) 
        prompt = PromptTemplate(template=config.SUMMARY_PROMPT, input_variables=["summary", "new_lines"])
        chain = prompt | components.llm | StrOutputParser()
        new_summary = chain.invoke({"summary": current_summary, "new_lines": older_messages})
        return {"summary": new_summary, "chat_history": history[-2:]}
    return {"summary": current_summary, "chat_history": history}

# --- LOGIC NODES ---

def route_query(state):
    print("---ROUTING---")
    question = state["question"]
    # If it's a Greeting or Code/Math, go General. Otherwise RAG.
    if any(x in question.lower() for x in ["hi", "hello", "code", "math", "write a function"]):
        return {"intent": "general"}
    return {"intent": "rag"}

def rewrite_query(state):
    print("---REWRITING---")
    question = state["question"]
    history = state.get("chat_history", [])

    # CRITICAL FIX: If no history, DO NOT REWRITE. 3B models hallucinate here.
    if not history:
        print("ℹ️ First turn: Skipping rewrite.")
        return {"standalone_question": question}

    context_str = f"Summary: {state.get('summary', '')}\nRecent: {history}"
    standalone = components.rewriter_chain.invoke({"history": context_str, "question": question})
    
    # Cleanup model output
    clean_q = standalone.split("Standalone Question:")[-1].strip()
    # If model failed to rewrite or returned garbage, fallback to original
    if len(clean_q) < 5: 
        clean_q = question
        
    print(f"📝 Rewritten: {clean_q}")
    return {"standalone_question": clean_q}

def retrieve(state):
    print("---RETRIEVING---")
    question = state.get("standalone_question", state["question"])
    
    # 1. Vector Search (Primary)
    documents = components.retriever.invoke(question)
    
    # 2. Graph Search (Secondary) - Boosts specific facts
    graph_context = search_graph(components.knowledge_graph, question)
    if graph_context:
        print(f"🕸️ Graph Found: {graph_context[:50]}...")
        graph_doc = Document(page_content=f"**KNOWLEDGE GRAPH FACT**:\n{graph_context}", metadata={"source": "graph"})
        documents.insert(0, graph_doc)
        
    # DEBUG: Print what we found so you know if retrieval is failing
    print(f"📄 Retrieved {len(documents)} docs. Top match: {documents[0].page_content[:100]}...")
    return {"documents": documents}

def generate_rag(state):
    print("---GENERATING (RAG)---")
    question = state["question"]
    documents = state["documents"]
    
    if not documents: 
        return {"generation": "I'm sorry, I couldn't find any information on that in the Shaastra documents."}

    doc_context = "\n\n".join([f"[Source: {d.metadata}]\n{d.page_content}" for d in documents])
    
    prompt = PromptTemplate(
        template=config.RAG_SYSTEM_PROMPT, 
        input_variables=["chat_history", "context", "question"]
    )
    
    # We removed 'summary' from prompt input to save context window on 3050 card
    chain = prompt | components.llm | StrOutputParser()
    gen = chain.invoke({
        "chat_history": "\n".join(state["chat_history"]), 
        "context": doc_context, 
        "question": question
    })
    
    if "Answer:" in gen: gen = gen.split("Answer:")[-1].strip()
    return {"generation": gen}

def generate_general(state):
    print("---GENERATING (GENERAL)---")
    # Simple prompt for general chat
    return {"generation": components.llm.invoke(state["question"])}