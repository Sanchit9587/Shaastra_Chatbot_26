# --- 1. CRITICAL FIX FOR CHROMA DB (SQLite FTS5) ---
try:
    __import__('pysqlite3')
    import sys
    sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')
except ImportError:
    pass
# ---------------------------------------------------

import sys
from typing import TypedDict, List
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langgraph.graph import END, StateGraph

# Import your setup functions
# Note: We now import create_hierarchical_retriever
from advanced_rag import create_hierarchical_retriever, load_llm, LLM_CONFIG, CONTEXT_FILE
from langchain_huggingface import HuggingFaceEmbeddings
import torch

# --- 1. SETUP STATE ---
class GraphState(TypedDict):
    question: str
    generation: str
    documents: List[str]

# --- 2. INITIALIZE MODELS ---

# Load Embedding
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Initializing Embeddings on {device}...")
embedding_model = HuggingFaceEmbeddings(model_name='all-MiniLM-L6-v2', model_kwargs={'device': device})

# ** Initialize Hierarchical Retriever **
retriever = create_hierarchical_retriever(CONTEXT_FILE, embedding_model)

# Load LLM
config = LLM_CONFIG["Unsloth 3B (Cached)"]
llm = load_llm(config["model_id"])

# --- 3. DEFINE NODES ---

def retrieve(state):
    """
    Retrieve documents (Parent Chunks) from vectorstore
    """
    print("---RETRIEVING (HIERARCHICAL)---")
    question = state["question"]
    
    # This invokes the ParentDocumentRetriever
    # It searches small chunks -> finds matches -> returns the LARGE parent chunks
    documents = retriever.invoke(question)
    
    return {"documents": documents, "question": question}

def grade_documents(state):
    """
    Basic check. In a full graph, you might use an LLM to grade here.
    For now, we trust the hierarchical retriever.
    """
    print("---CHECKING RETRIEVED DOCS---")
    documents = state["documents"]
    
    if documents:
        print(f"---FOUND {len(documents)} RELEVANT PARENT CONTEXTS---")
        # Optional: Print first few chars to debug context
        # print(f"Preview: {documents[0].page_content[:200]}...")
    else:
        print("---NO RELEVANT DOCUMENTS FOUND---")
            
    return {"documents": documents, "question": state["question"]}

def generate(state):
    """
    Generate answer using RAG on documents
    """
    print("---GENERATING---")
    question = state["question"]
    documents = state["documents"]
    
    # Format context
    context = "\n\n".join([doc.page_content for doc in documents])
    
    # Use prompt template
    prompt = PromptTemplate(
        template=config["prompt_template"],
        input_variables=["context", "question"]
    )
    
    chain = prompt | llm | StrOutputParser()
    generation = chain.invoke({"context": context, "question": question})
    return {"documents": documents, "question": question, "generation": generation}

# --- 4. BUILD GRAPH ---

workflow = StateGraph(GraphState)

workflow.add_node("retrieve", retrieve)
workflow.add_node("grade_documents", grade_documents)
workflow.add_node("generate", generate)

workflow.set_entry_point("retrieve")
workflow.add_edge("retrieve", "grade_documents")

def decide_to_generate(state):
    if not state["documents"]:
        return "end_no_data"
    return "generate"

workflow.add_conditional_edges(
    "grade_documents",
    decide_to_generate,
    {
        "generate": "generate",
        "end_no_data": END
    }
)

workflow.add_edge("generate", END)
app = workflow.compile()

# --- 5. EXECUTION ---

if __name__ == "__main__":
    while True:
        user_query = input("\nLangGraph Hierarchical RAG - Ask a question (or 'exit'): ")
        if user_query.lower() == 'exit':
            break
            
        inputs = {"question": user_query}
        try:
            final_result = app.invoke(inputs)
            
            if not final_result["documents"] and "generation" not in final_result:
                 print("\nAnswer: I do not have enough relevant context to answer this.")
            else:
                 print(f"\nAnswer: {final_result['generation']}")
        except Exception as e:
            print(f"Error executing graph: {e}")
            import traceback
            traceback.print_exc()