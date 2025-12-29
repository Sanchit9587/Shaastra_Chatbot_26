import networkx as nx
import re
import shutil
import os
from langchain_community.document_loaders import TextLoader
from langchain.storage import InMemoryStore
from langchain_text_splitters import RecursiveCharacterTextSplitter, MarkdownHeaderTextSplitter
from langchain_chroma import Chroma
from langchain.retrievers import ParentDocumentRetriever
import config

# Synonyms to link distinct concepts in the Graph
SYNONYMS = {
    "food": ["Himalaya Food Court", "Campus Cafe", "Zaitoon", "Mess"],
    "eat": ["Himalaya Food Court", "Campus Cafe"],
    "stay": ["Accommodation", "Hostel"],
    "sleep": ["Accommodation", "Hostel"],
    "hackathon": ["IndustriAI", "FedEx SMART", "Appian AI"],
    "coding": ["Shaastra Programming Contest", "Clash of Codes", "Reverse Coding"],
    "robot": ["Robowars", "Robosoccer", "Caterpillar Autonomy"]
}

def build_knowledge_graph(docs):
    print("Building Knowledge Graph...")
    G = nx.Graph()
    text = docs[0].page_content
    lines = text.split('\n')
    
    current_section = "General"
    
    for line in lines:
        line = line.strip()
        if not line: continue
        
        # Track Sections
        if line.startswith("## "):
            current_section = line.replace("## ", "").strip()
            G.add_node(current_section, type="Event")
        
        # 1. Table Parsing
        if "|" in line and "Event" not in line and "---" not in line:
            parts = [p.strip() for p in line.split('|') if p.strip()]
            if len(parts) >= 2:
                event = parts[0]
                venue = parts[1]
                G.add_edge(event, venue, relation="hosted_at")
                if len(parts) >= 3:
                    G.add_edge(event, parts[2], relation="happens_at")

        # 2. Key-Value Extraction
        match_venue = re.search(r'(?i)\*?Venue:?\*?\s*(.*)', line)
        if match_venue:
            venue = match_venue.group(1).strip()
            G.add_edge(current_section, venue, relation="hosted_at")
            
        match_date = re.search(r'(?i)\*?Date:?\*?\s*(.*)', line)
        if match_date:
            date_info = match_date.group(1).strip()
            G.add_edge(current_section, date_info, relation="happens_on")

    return G

def search_graph(G, query):
    context_strings = []
    query_lower = query.lower()
    
    search_terms = [query_lower]
    for key, values in SYNONYMS.items():
        if key in query_lower:
            search_terms.extend([v.lower() for v in values])
            
    found_nodes = set()
    for node in G.nodes():
        node_lower = node.lower()
        if any(term in node_lower for term in search_terms):
            found_nodes.add(node)
            
    for node in found_nodes:
        neighbors = list(G.neighbors(node))
        if neighbors:
            context_strings.append(f"{node} is related to: {', '.join(neighbors)}.")
            
    return "\n".join(context_strings)

def create_retrieval_engines(file_path, embedding_model):
    print("Loading Document...")
    loader = TextLoader(file_path)
    docs = loader.load()

    kg = build_knowledge_graph(docs)

    # Improved Splitter Strategy
    md_splitter = MarkdownHeaderTextSplitter(headers_to_split_on=[
        ("#", "Header 1"), ("##", "Header 2"), ("**", "Header 3")
    ])
    parent_docs = md_splitter.split_text(docs[0].page_content)

    child_splitter = RecursiveCharacterTextSplitter(chunk_size=400, chunk_overlap=50)
    
    # --- CRITICAL FIX: CLEAN START EVERY TIME ---
    # We force delete the collection to ensure InMemoryStore and Chroma are in sync.
    if os.path.exists(config.CHROMA_DB_DIR):
        try:
            shutil.rmtree(config.CHROMA_DB_DIR)
            print("🧹 Cleared old vector database cache.")
        except Exception as e:
            print(f"⚠️ Could not clear DB: {e}")

    vectorstore = Chroma(
        collection_name="shaastra_prod", 
        embedding_function=embedding_model,
        persist_directory=str(config.CHROMA_DB_DIR)
    )
    store = InMemoryStore()

    retriever = ParentDocumentRetriever(
        vectorstore=vectorstore, 
        docstore=store, 
        child_splitter=child_splitter,
        parent_splitter=RecursiveCharacterTextSplitter(chunk_size=1200),
        search_kwargs={"k": 4}
    )
    
    print("Indexing Documents...")
    retriever.add_documents(parent_docs, ids=None)

    return retriever, kg