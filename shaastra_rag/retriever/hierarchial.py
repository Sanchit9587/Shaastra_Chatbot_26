from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import MarkdownHeaderTextSplitter, RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain.storage import InMemoryStore
from langchain.retrievers import ParentDocumentRetriever
from .knowledge_graph import build_knowledge_graph
from .. import config

def create_hierarchical_retriever(file_path, embedding_model):
    print(f"Loading document from {file_path}...")
    loader = TextLoader(str(file_path))
    docs = loader.load()

    # Build Graph
    kg = build_knowledge_graph(docs)

    # Markdown Splitter (Parent Chunks)
    headers_to_split_on = [
        ("#", "Header 1"),
        ("##", "Header 2"),
        ("###", "Header 3"),
    ]
    md_splitter = MarkdownHeaderTextSplitter(headers_to_split_on=headers_to_split_on)
    parent_docs = md_splitter.split_text(docs[0].page_content)

    # Child Splitter
    child_splitter = RecursiveCharacterTextSplitter(chunk_size=400, chunk_overlap=50)

    # Vectorstore
    vectorstore = Chroma(
        collection_name="shaastra_hierarchical", 
        embedding_function=embedding_model,
        persist_directory=str(config.CHROMA_DB_DIR) 
    )

    store = InMemoryStore()

    print("Creating Hierarchical Retriever...")
    retriever = ParentDocumentRetriever(
        vectorstore=vectorstore,
        docstore=store,
        child_splitter=child_splitter,
        parent_splitter=RecursiveCharacterTextSplitter(chunk_size=2000),
        search_kwargs={"k": 20} # Retrieve broad candidates for reranker
    )

    print(f"Indexing {len(parent_docs)} parent sections...")
    retriever.add_documents(parent_docs, ids=None)
    
    return retriever, kg