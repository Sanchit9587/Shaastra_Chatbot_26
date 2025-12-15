from typing import TypedDict, List, Any
from langgraph.graph import END, StateGraph
import nodes

class GraphState(TypedDict):
    question: str
    standalone_question: str
    generation: str
    documents: List[Any]
    chat_history: List[str]
    summary: str # <--- Added Summary State
    intent: str

def build_graph():
    workflow = StateGraph(GraphState)
    
    # Nodes
    workflow.add_node("summarize", nodes.summarize_conversation) # New Memory Manager
    workflow.add_node("router", nodes.route_query)
    workflow.add_node("rewrite", nodes.rewrite_query)
    workflow.add_node("retrieve", nodes.retrieve)
    workflow.add_node("generate_rag", nodes.generate_rag)
    workflow.add_node("generate_general", nodes.generate_general)
    
    # Entry Point: Always try to summarize old memory first
    workflow.set_entry_point("summarize")
    
    # Flow
    workflow.add_edge("summarize", "router")
    
    def route_decision(state): return state["intent"]
    
    workflow.add_conditional_edges("router", route_decision, {
        "rag": "rewrite", # Rewrite is always good for RAG
        "general": "generate_general"
    })
    
    workflow.add_edge("rewrite", "retrieve")
    workflow.add_edge("retrieve", "generate_rag")
    workflow.add_edge("generate_rag", END)
    workflow.add_edge("generate_general", END)
    
    return workflow.compile()