# chains/query_rewriter.py
from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

def create_query_rewriter(llm):
    template = """"""  # (exact same as your original query_transform.py)
    prompt = PromptTemplate(template=template, input_variables=["history", "question"])
    return prompt | llm | StrOutputParser()