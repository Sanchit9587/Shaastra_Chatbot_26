from typing import TypedDict, List, Any
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.documents import Document
from langgraph.graph import END, StateGraph

from .. import config
from ..utils import format_history, print_retrieved_chunks
from ..retriever.knowledge_graph import search_graph

# State Definition
class GraphState(TypedDict):
    question: str
    standalone_question: str
    generation: str
    documents: List[Any]
    chat_history: List[str]
    intent: str

# Class to hold loaded models (Singleton Pattern)
class RAGContext:
    def __init__(self, llm, retriever, kg, rewriter_chain, grader_chain):
        self.llm = llm
        self.retriever = retriever
        self.kg = kg
        self.rewriter = rewriter_chain
        self.grader = grader_chain

# Nodes Logic
def route_query(state, context: RAGContext):
    print("---ROUTING---")
    question = state["question"]
    history = format_history(state["chat_history"])
    
    pronouns = ['it', 'that', 'this', 'he', 'she', 'they']
    if any(word in question.lower().split() for word in pronouns):
        print("---DECISION: FOLLOWUP---")
        return {"intent": "followup"}

    prompt = PromptTemplate(template=config.ROUTER_PROMPT, input_variables=["history", "question"])
    chain = prompt | context.llm | StrOutputParser()
    decision = chain.invoke({"history": history, "question": question}).strip().lower()
    
    print(f"---DECISION: {decision.upper()}---")
    if "rag" in decision: return {"intent": "rag"}
    return {"intent": "general"}

def rewrite_query(state, context: RAGContext):
    print("---REWRITING---")
    question = state["question"]
    history = format_history(state["chat_history"])
    if not history: return {"standalone_question": question}
    
    standalone = context.rewriter.invoke({"history": history, "question": question})
    if "Standalone Question:" in standalone:
        standalone = standalone.split("Standalone Question:")[-1].strip()
    print(f"Rewritten: {standalone}")
    return {"standalone_question": standalone}

def retrieve(state, context: RAGContext):
    print("---RETRIEVING---")
    question = state.get("standalone_question", state["question"])
    
    # Hybrid Search
    documents = context.retriever.invoke(question)
    
    # Graph Search
    graph_context = search_graph(context.kg, question)
    if graph_context:
        print("🕸️ Graph Context Injected")
        graph_doc = Document(
            page_content=f"**VERIFIED GRAPH FACTS**:\n{graph_context}", 
            metadata={"Header 1": "Knowledge Graph"}
        )
        documents.insert(0, graph_doc)
        
    print_retrieved_chunks(documents)
    return {"documents": documents}

def generate_rag(state, context: RAGContext):
    print("---GENERATING (RAG)---")
    question = state["question"]
    documents = state["documents"]
    
    if not documents: return {"generation": "I couldn't find info on that."}

    doc_context = "\n".join([f"[Header: {d.metadata}]\n{d.page_content}" for d in documents])
    prompt = PromptTemplate(template=config.RAG_SYSTEM_PROMPT, input_variables=["chat_history", "context", "question"])
    
    chain = prompt | context.llm | StrOutputParser()
    gen = chain.invoke({
        "chat_history": format_history(state["chat_history"]), 
        "context": doc_context, 
        "question": question
    })
    return {"generation": gen}

def grade_generation(state, context: RAGContext):
    print("---GRADING---")
    if not state.get("documents"): return {"generation": state["generation"]}
    
    doc_text = "\n".join([d.page_content for d in state["documents"]])
    score = context.grader.invoke({"documents": doc_text, "generation": state["generation"]})
    
    if "no" in score.lower():
        print("❌ Hallucination Detected")
        return {"generation": "I apologize, I could not find verified details in the official documents."}
    
    print("✅ Answer Validated")
    return {"generation": state["generation"]}

def generate_general(state, context: RAGContext):
    print("---GENERATING (GENERAL)---")
    prompt = PromptTemplate(template="<|begin_of_text|><|start_header_id|>system<|end_header_id|>You are a helpful AI.<|eot_id|><|start_header_id|>user<|end_header_id|>{q}<|eot_id|><|start_header_id|>assistant<|end_header_id|>", input_variables=["q"])
    chain = prompt | context.llm | StrOutputParser()
    return {"generation": chain.invoke({"q": state["question"]})}

def build_app(context: RAGContext):
    workflow = StateGraph(GraphState)
    
    # Register Nodes with Context using Lambda
    workflow.add_node("router", lambda state: route_query(state, context))
    workflow.add_node("rewrite", lambda state: rewrite_query(state, context))
    workflow.add_node("retrieve", lambda state: retrieve(state, context))
    workflow.add_node("generate_rag", lambda state: generate_rag(state, context))
    workflow.add_node("grader", lambda state: grade_generation(state, context))
    workflow.add_node("generate_general", lambda state: generate_general(state, context))
    
    workflow.set_entry_point("router")
    
    def route_decision(state): return state["intent"]
    
    workflow.add_conditional_edges("router", route_decision, {
        "rag": "retrieve", "followup": "rewrite", "general": "generate_general"
    })
    
    workflow.add_edge("rewrite", "retrieve")
    workflow.add_edge("retrieve", "generate_rag")
    workflow.add_edge("generate_rag", "grader")
    workflow.add_edge("grader", END)
    workflow.add_edge("generate_general", END)
    
    return workflow.compile()