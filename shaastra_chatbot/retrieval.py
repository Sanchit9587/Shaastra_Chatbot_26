import networkx as nx
import re
from langchain_community.document_loaders import TextLoader
from langchain.storage import InMemoryStore
from langchain_text_splitters import RecursiveCharacterTextSplitter, MarkdownHeaderTextSplitter
from langchain_chroma import Chroma
from langchain.retrievers import ParentDocumentRetriever
import config

def build_knowledge_graph(docs):
    print("Building Knowledge Graph...")
    G = nx.Graph()
    text = docs[0].page_content
    lines = text.split('\n')
    
    current_event = None
    
    for line in lines:
        line = line.strip()
        if not line: continue
        
        # 1. Table Parsing (Existing)
        if "|" in line and "Event" not in line and "---" not in line:
            parts = [p.strip() for p in line.split('|') if p.strip()]
            if len(parts) >= 2:
                # Event -> Venue
                G.add_edge(parts[0], parts[1], relation="hosted_at")
                if len(parts) >= 3:
                     # Event -> Time
                    G.add_edge(parts[0], parts[2], relation="happens_at")

        # 2. Header Parsing (New: catches "## Moot Court")
        if line.startswith("## "):
            current_event = line.replace("## ", "").strip()
        
        # 3. Key-Value Extraction from Paragraphs
        # Looks for "Venue: OAT" or "Prize Pool: 85000"
        if current_event:
            # Check for Venue
            if "Venue" in line or "at " in line:
                # Simple heuristic to extract potential venues
                keywords = ["OAT", "CRC", "SAC", "CLT", "KV", "ICSR", "RJN", "RMN"]
                for k in keywords:
                    if k in line:
                        G.add_edge(current_event, k, relation="hosted_at")
            
            # Check for Dates
            if "January" in line:
                 G.add_edge(current_event, line, relation="happens_on")

    return G

def search_graph(G, query):
    context_strings = []
    query_lower = query.lower()
    
    # Fuzzy match node names
    for node in G.nodes():
        if node.lower() in query_lower:
            neighbors = list(G.neighbors(node))
            if neighbors:
                context_strings.append(f"{node} is associated with: {', '.join(neighbors)}.")
    
    return "\n".join(context_strings)

def create_retrieval_engines(file_path, embedding_model):
    print("Loading Document...")
    loader = TextLoader(file_path)
    docs = loader.load()

    # 1. Build Graph
    kg = build_knowledge_graph(docs)

    # 2. Markdown Splitter - CRITICAL FOR RETRIEVAL
    # We split by Headers so "Moot Court" gets its own chunk
    md_splitter = MarkdownHeaderTextSplitter(headers_to_split_on=[
        ("#", "Header 1"), ("##", "Header 2"), ("**", "Header 3")
    ])
    parent_docs = md_splitter.split_text(docs[0].page_content)

    # 3. Vectorstore (Chroma)
    # Using smaller chunks for child documents to hit specific facts like "Prize Pool"
    child_splitter = RecursiveCharacterTextSplitter(chunk_size=300, chunk_overlap=50)
    
    vectorstore = Chroma(
        collection_name="shaastra_prod", 
        embedding_function=embedding_model,
        persist_directory="./chroma_db_prod" 
    )
    store = InMemoryStore()

    retriever = ParentDocumentRetriever(
        vectorstore=vectorstore, 
        docstore=store, 
        child_splitter=child_splitter,
        parent_splitter=RecursiveCharacterTextSplitter(chunk_size=1000),
        search_kwargs={"k": 5} # Fetch top 5 relevant docs
    )
    
    print("Indexing Documents...")
    retriever.add_documents(parent_docs, ids=None)
    
    # Removed Reranker for speed/stability on 3050 - Vector Search is usually enough if chunking is good
    return retriever, kg