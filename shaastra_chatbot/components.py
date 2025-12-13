# components.py
import config
from models import load_llm, load_embedding_model
from retrieval import create_retrieval_engines
from chains import create_query_rewriter

# Class to hold loaded models (Singleton Pattern)
class RAGContext:
    def __init__(self, llm, retriever, kg, rewriter_chain):
        self.llm = llm
        self.retriever = retriever
        self.kg = kg
        self.rewriter = rewriter_chain

# Load Global Components
llm = load_llm(config.LLM_MODEL_ID)
embedding_model = load_embedding_model(config.EMBEDDING_MODEL_ID)
retriever, knowledge_graph = create_retrieval_engines(config.CONTEXT_FILE, embedding_model)

# Load Chains (Removed Grader)
rewriter_chain = create_query_rewriter(llm)

# Initialize Context
context = RAGContext(llm, retriever, knowledge_graph, rewriter_chain)