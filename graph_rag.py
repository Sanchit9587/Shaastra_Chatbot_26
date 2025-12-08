import sys
import operator
from typing import Annotated, Sequence, TypedDict, List
from typing_extensions import TypedDict

# LangChain & LangGraph Imports
from langchain_core.messages import BaseMessage, HumanMessage
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser, StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langgraph.graph import END, StateGraph

# Import your existing setup functions
# (Assuming your advanced_rag.py functions are importable or pasted here)
# For this example, I will assume we use the components defined in your previous code.
from advanced_rag import load_and_chunk_document, load_llm, LLM_CONFIG
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

# --- 1. SETUP STATE ---
class GraphState(TypedDict):
    """
    Represents the state of our graph.
    
    Attributes:
        question: question
        generation: LLM generation
        documents: list of documents
    """
    question: str
    generation: str
    documents: List[str]

# --- 2. INITIALIZE MODELS (Reusing your logic) ---
# Load Context
CONTEXT_FILE = "rag_context.md"
chunks = load_and_chunk_document(CONTEXT_FILE)

# Load Embedding
device = 'cuda' # or 'cpu'
embedding_model = HuggingFaceEmbeddings(model_name='all-MiniLM-L6-v2', model_kwargs={'device': device})

# Load Vector Store (Simple Chroma for graph demonstration)
vectorstore = Chroma.from_documents(chunks, embedding_model)
retriever = vectorstore.as_retriever(search_kwargs={"k": 5})

# Load LLM (Using your Unsloth setup)
config = LLM_CONFIG["Unsloth 3B (Cached)"]
llm = load_llm(config["model_id"])

# --- 3. DEFINE NODES ---

def retrieve(state):
    """
    Retrieve documents from vectorstore
    """
    print("---RETRIEVING---")
    question = state["question"]
    documents = retriever.invoke(question)
    return {"documents": documents, "question": question}

def grade_documents(state):
    """
    Determines whether the retrieved documents are relevant to the question
    If any document is not relevant, we filter it out.
    """
    print("---CHECKING RELEVANCE---")
    question = state["question"]
    documents = state["documents"]
    
    # Prompt for grading
    prompt = PromptTemplate(
        template="""You are a grader assessing relevance of a retrieved document to a user question. \n 
        Here is the retrieved document: \n\n {context} \n\n
        Here is the user question: {question} \n
        If the document contains keywords or semantic meaning related to the question, grade it as relevant. \n
        Give a binary score 'yes' or 'no' score to indicate whether the document is relevant to the question. \n
        Return only 'yes' or 'no'.""",
        input_variables=["context", "question"],
    )
    
    chain = prompt | llm | StrOutputParser()
    
    filtered_docs = []
    for d in documents:
        score = chain.invoke({"question": question, "context": d.page_content})
        # Simple parsing logic (LLM might be chatty, checking for 'yes')
        if "yes" in score.lower():
            print("---GRADE: DOCUMENT RELEVANT---")
            filtered_docs.append(d)
        else:
            print("---GRADE: DOCUMENT NOT RELEVANT---")
            continue
            
    return {"documents": filtered_docs, "question": question}

def generate(state):
    """
    Generate answer using RAG on filtered documents
    """
    print("---GENERATING---")
    question = state["question"]
    documents = state["documents"]
    
    # Format context
    context = "\n\n".join([doc.page_content for doc in documents])
    
    # Use your specific prompt template
    prompt = PromptTemplate(
        template=config["prompt_template"],
        input_variables=["context", "question"]
    )
    
    chain = prompt | llm | StrOutputParser()
    generation = chain.invoke({"context": context, "question": question})
    return {"documents": documents, "question": question, "generation": generation}

# --- 4. BUILD GRAPH ---

workflow = StateGraph(GraphState)

# Define the nodes
workflow.add_node("retrieve", retrieve)
workflow.add_node("grade_documents", grade_documents)
workflow.add_node("generate", generate)

# Build the edges
workflow.set_entry_point("retrieve")
workflow.add_edge("retrieve", "grade_documents")

# Conditional Edge Logic
def decide_to_generate(state):
    """
    Determines whether to generate an answer, or re-generate a question (if no docs found).
    For simplicity in this V1, if no docs are relevant, we just say 'I don't know'.
    In V2, you would add a 'transform_query' node here.
    """
    filtered_documents = state["documents"]
    if not filtered_documents:
        # If no relevant documents found, goes to end (or could go to web search)
        print("---DECISION: NO RELEVANT DOCUMENTS FOUND---")
        return "end_no_data"
    else:
        print("---DECISION: GENERATE---")
        return "generate"

# Add conditional edges
workflow.add_conditional_edges(
    "grade_documents",
    decide_to_generate,
    {
        "generate": "generate",
        "end_no_data": END # This stops the graph
    }
)

workflow.add_edge("generate", END)

# Compile
app = workflow.compile()

# --- 5. EXECUTION ---

if __name__ == "__main__":
    while True:
        user_query = input("\nLangGraph RAG - Ask a question (or 'exit'): ")
        if user_query.lower() == 'exit':
            break
            
        inputs = {"question": user_query}
        for output in app.stream(inputs):
            for key, value in output.items():
                print(f"Finished Node: {key}")
                
        # Final result is typically in the last state
        # Because we stream, we might need to capture the final result differently
        # Or just run invoke for the final answer:
        final_result = app.invoke(inputs)
        
        if not final_result["documents"] and "generation" not in final_result:
             print("\nAnswer: I do not have enough relevant context to answer this.")
        else:
             print(f"\nAnswer: {final_result['generation']}")