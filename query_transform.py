# query_transform.py
from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

def create_query_rewriter(llm):
    """
    Creates a chain that rewrites the user query to be standalone
    based on chat history.
    """
    
    rewrite_prompt = PromptTemplate(
        template="""<|begin_of_text|><|start_header_id|>system<|end_header_id|>
You are a helpful assistant that reformulates questions.
Given a chat history and the latest user question, rephrase the question to be a standalone question that can be understood without the history.
Do NOT answer the question. Just rewrite it.
If the question is already standalone, return it exactly as is.

Chat History:
{history}

Latest Question: {question}<|eot_id|><|start_header_id|>assistant<|end_header_id|>
Standalone Question:""",
        input_variables=["history", "question"]
    )

    return rewrite_prompt | llm | StrOutputParser()