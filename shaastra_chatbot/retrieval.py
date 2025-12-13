# retrieval.py
import networkx as nx
from langchain_community.document_loaders import TextLoader
from langchain.storage import InMemoryStore
from langchain_text_splitters import RecursiveCharacterTextSplitter, MarkdownHeaderTextSplitter
from langchain_chroma import Chroma
from langchain.retrievers import ParentDocumentRetriever, ContextualCompressionRetriever
from langchain.retrievers.document_compressors import CrossEncoderReranker
from langchain_community.cross_encoders import HuggingFaceCrossEncoder
import config

def build_knowledge_graph(docs):
    print("Building Knowledge Graph...")
    G = nx.Graph()
    for doc in docs:
        content = doc.page_content
        lines = content.split('\n')
        for line in lines:
            if "|" in line and "Event" not in line and "---" not in line:
                parts = [p.strip() for p in line.split('|') if p.strip()]
                if len(parts) >= 2:
                    G.add_edge(parts[0], parts[1], relation="hosted_at")
    return G

def search_graph(G, query):
    context_strings = []
    query_lower = query.lower()
    for node in G.nodes():
        if node.lower() in query_lower:
            neighbors = list(G.neighbors(node))
            if neighbors:
                context_strings.append(f"GRAPH FACT: {node} is connected to {', '.join(neighbors)}.")
    return "\n".join(context_strings)

def create_retrieval_engines(file_path, embedding_model):
    print("Loading Document...")
    loader = TextLoader(file_path)
    docs = loader.load()

    # 1. Build Graph
    kg = build_knowledge_graph(docs)

    # 2. Markdown Splitter
    md_splitter = MarkdownHeaderTextSplitter(headers_to_split_on=[
        ("#", "Header 1"), ("##", "Header 2"), ("###", "Header 3")
    ])
    parent_docs = md_splitter.split_text(docs[0].page_content)

    # 3. Vectorstore
    child_splitter = RecursiveCharacterTextSplitter(chunk_size=400, chunk_overlap=50)
    vectorstore = Chroma(
        collection_name="shaastra_prod", 
        embedding_function=embedding_model,
        persist_directory="./chroma_db_prod" 
    )
    store = InMemoryStore()

    base_retriever = ParentDocumentRetriever(
        vectorstore=vectorstore, docstore=store, child_splitter=child_splitter,
        parent_splitter=RecursiveCharacterTextSplitter(chunk_size=2000),
        search_kwargs={"k": 20}
    )
    
    print("Indexing Documents...")
    base_retriever.add_documents(parent_docs, ids=None)

    # 4. Reranker
    print(f"Loading Reranker ({config.RERANKER_MODEL_ID})...")
    rerank_model = HuggingFaceCrossEncoder(model_name=config.RERANKER_MODEL_ID)
    compressor = CrossEncoderReranker(model=rerank_model, top_n=5)
    
    final_retriever = ContextualCompressionRetriever(
        base_compressor=compressor, base_retriever=base_retriever
    )

    return final_retriever, kg