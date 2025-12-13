# nodes.py
from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.documents import Document
import config
import components 
from retrieval import search_graph

def format_history(history):
    return "\n".join(history[-6:]) 

def route_query(state):
    print("---ROUTING---")
    question = state["question"]
    history = format_history(state["chat_history"])
    
    pronouns = ['it', 'that', 'this', 'he', 'she', 'they']
    if any(word in question.lower().split() for word in pronouns):
        return {"intent": "followup"}

    prompt = PromptTemplate(template=config.ROUTER_PROMPT, input_variables=["history", "question"])
    chain = prompt | components.llm | StrOutputParser()
    decision = chain.invoke({"history": history, "question": question}).strip().lower()
    
    if "rag" in decision: return {"intent": "rag"}
    return {"intent": "general"}

def rewrite_query(state):
    print("---REWRITING---")
    question = state["question"]
    history = format_history(state["chat_history"])
    if not history: return {"standalone_question": question}
    
    standalone = components.rewriter_chain.invoke({"history": history, "question": question})
    if "Standalone Question:" in standalone:
        standalone = standalone.split("Standalone Question:")[-1].strip()
    return {"standalone_question": standalone}

def retrieve(state):
    print("---RETRIEVING---")
    question = state.get("standalone_question", state["question"])
    documents = components.retriever.invoke(question)
    
    graph_context = search_graph(components.knowledge_graph, question)
    if graph_context:
        print("🕸️ Graph Context Found")
        graph_doc = Document(page_content=f"**GRAPH FACT**:\n{graph_context}", metadata={"Header 1": "Graph"})
        documents.insert(0, graph_doc)
        
    return {"documents": documents}

def generate_rag(state):
    print("---GENERATING (RAG + CoT)---")
    question = state["question"]
    documents = state["documents"]
    
    if not documents: return {"generation": "I'm sorry, I couldn't find any information about that in the Shaastra documents."}

    context = "\n\n".join([f"[Header: {d.metadata}]\n{d.page_content}" for d in documents])
    prompt = PromptTemplate(template=config.SYSTEM_PROMPT, input_variables=["chat_history", "context", "question"])
    chain = prompt | components.llm | StrOutputParser()
    
    gen = chain.invoke({
        "chat_history": format_history(state["chat_history"]), 
        "context": context, "question": question
    })
    return {"generation": gen}

def generate_general(state):
    print("---GENERATING (GENERAL)---")
    prompt = PromptTemplate(template="<|begin_of_text|><|start_header_id|>system<|end_header_id|>Helpful AI.<|eot_id|><|start_header_id|>user<|end_header_id|>{q}<|eot_id|><|start_header_id|>assistant<|end_header_id|>", input_variables=["q"])
    chain = prompt | components.llm | StrOutputParser()
    return {"generation": chain.invoke({"q": state["question"]})}