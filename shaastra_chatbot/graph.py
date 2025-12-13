# graph.py
from typing import TypedDict, List, Any
from langgraph.graph import END, StateGraph
import nodes

class GraphState(TypedDict):
    question: str
    standalone_question: str
    generation: str
    documents: List[Any]
    chat_history: List[str]
    intent: str

def build_graph():
    workflow = StateGraph(GraphState)
    
    workflow.add_node("router", nodes.route_query)
    workflow.add_node("rewrite", nodes.rewrite_query)
    workflow.add_node("retrieve", nodes.retrieve)
    workflow.add_node("generate_rag", nodes.generate_rag)
    # Removed grader node
    workflow.add_node("generate_general", nodes.generate_general)
    
    workflow.set_entry_point("router")
    
    def route_decision(state): return state["intent"]
    
    workflow.add_conditional_edges("router", route_decision, {
        "rag": "retrieve", "followup": "rewrite", "general": "generate_general"
    })
    
    workflow.add_edge("rewrite", "retrieve")
    workflow.add_edge("retrieve", "generate_rag")
    # Changed edge: Rag generation now goes straight to End
    workflow.add_edge("generate_rag", END)
    workflow.add_edge("generate_general", END)
    
    return workflow.compile()