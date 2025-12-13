from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

def create_grader(llm):
    grader_prompt = PromptTemplate(
        template="""<|begin_of_text|><|start_header_id|>system<|end_header_id|>
You are a lenient grader assessing an AI generation.
Check if the generated answer is grounded in / supported by the provided facts.

- If the answer is correct based on the facts, output 'yes'.
- If the answer is mostly correct but adds minor polite text, output 'yes'.
- Only output 'no' if the answer is completely wrong or contradicts the facts.

FACTS:
{documents}

GENERATED ANSWER:
{generation}
<|eot_id|><|start_header_id|>assistant<|end_header_id|>
Supported (yes/no):""",
        input_variables=["documents", "generation"]
    )
    return grader_prompt | llm | StrOutputParser()