# chains.py
from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

def create_query_rewriter(llm):
    prompt = PromptTemplate(
        template="""<|begin_of_text|><|start_header_id|>system<|end_header_id|>
Rewrite the user question to be standalone based on history.
History: {history}
Question: {question}<|eot_id|><|start_header_id|>assistant<|end_header_id|>
Standalone Question:""",
        input_variables=["history", "question"]
    )
    return prompt | llm | StrOutputParser()

def create_grader(llm):
    prompt = PromptTemplate(
        template="""<|begin_of_text|><|start_header_id|>system<|end_header_id|>
You are a lenient grader. Check if the answer is grounded in facts.
Output 'yes' if correct or mostly correct. Output 'no' ONLY if completely wrong.
Facts: {documents}
Answer: {generation}<|eot_id|><|start_header_id|>assistant<|end_header_id|>
Supported (yes/no):""",
        input_variables=["documents", "generation"]
    )
    return prompt | llm | StrOutputParser()