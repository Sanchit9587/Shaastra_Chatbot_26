from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, pipeline
from langchain_huggingface import HuggingFaceEmbeddings, HuggingFacePipeline
import torch
from .utils import check_gpu_status

# Types for Pydantic Fix
from typing import Union, List, Optional, Any, Dict
from langchain_core.caches import BaseCache
from langchain_core.callbacks import Callbacks, BaseCallbackHandler
from langchain_core.language_models import BaseLanguageModel

def load_embedding_model(model_id):
    device = check_gpu_status()
    print(f"Loading Embeddings ({model_id})...")
    return HuggingFaceEmbeddings(model_name=model_id, model_kwargs={'device': device})

def load_llm(model_id):
    print(f"Loading LLM ({model_id})...")
    
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
    )
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        quantization_config=bnb_config,
        device_map="auto",
        torch_dtype=torch.float16,
        trust_remote_code=True
    )
    hf_pipeline = pipeline(
        "text-generation", 
        model=model, 
        tokenizer=tokenizer, 
        max_new_tokens=512,
        return_full_text=False
    )

    # --- CRITICAL PYDANTIC FIX ---
    try:
        HuggingFacePipeline.model_rebuild(_types_namespace={
            "Union": Union, "List": List, "Optional": Optional, "Dict": Dict, "Any": Any,
            "Callbacks": Callbacks, "BaseCallbackHandler": BaseCallbackHandler,
            "BaseCache": BaseCache, "BaseLanguageModel": BaseLanguageModel
        })
    except Exception:
        pass
    
    return HuggingFacePipeline(pipeline=hf_pipeline)