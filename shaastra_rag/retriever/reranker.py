from langchain_community.cross_encoders import HuggingFaceCrossEncoder
from langchain.retrievers.document_compressors import CrossEncoderReranker
from langchain.retrievers import ContextualCompressionRetriever
from .. import config

def add_reranker(base_retriever):
    print(f"Loading Reranker ({config.RERANKER_MODEL_ID})...")
    
    model = HuggingFaceCrossEncoder(model_name=config.RERANKER_MODEL_ID)
    
    # Keep top 5 most relevant chunks
    compressor = CrossEncoderReranker(model=model, top_n=5)
    
    compression_retriever = ContextualCompressionRetriever(
        base_compressor=compressor, 
        base_retriever=base_retriever
    )
    
    return compression_retriever