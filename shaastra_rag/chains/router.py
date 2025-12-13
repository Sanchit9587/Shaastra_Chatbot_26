# chains/router.py
from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

def create_router_chain(llm):
    template = """..."""  # Use the smart router prompt from graph_rag2.py
    prompt = PromptTemplate(template=template, input_variables=["history", "question"])
    return prompt | llm | StrOutputParser()