# grader.py
from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

def create_grader(llm):
    """
    Creates a chain that checks if the generated answer is supported by the context.
    """
    
    grader_prompt = PromptTemplate(
        template="""<|begin_of_text|><|start_header_id|>system<|end_header_id|>
You are a grader assessing an AI generation.
Check if the generated answer is grounded in / supported by the provided facts.
Give a binary score 'yes' or 'no'.
'yes' means the answer is supported by the facts.
'no' means the answer contains information NOT found in the facts.

FACTS:
{documents}

GENERATED ANSWER:
{generation}
<|eot_id|><|start_header_id|>assistant<|end_header_id|>
Supported:""",
        input_variables=["documents", "generation"]
    )

    return grader_prompt | llm | StrOutputParser()