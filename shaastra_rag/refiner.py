from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

def clean_scraped_data(raw_text, llm):
    """
    Uses the loaded LLM to summarize and clean web data.
    """
    print("🧠 Refining text with LLM...")
    
    prompt = PromptTemplate(
        template="""<|begin_of_text|><|start_header_id|>system<|end_header_id|>
You are an expert data cleaner.
1. Read the raw text below scraped from a website.
2. Remove navigation menus, copyright footers, and code snippets.
3. Organize the key information into clear Markdown sections.
4. Keep specific details like Dates, Venues, and Rules intact.

RAW TEXT:
{raw_text}
<|eot_id|><|start_header_id|>assistant<|end_header_id|>
Cleaned Markdown Summary:""",
        input_variables=["raw_text"]
    )
    
    chain = prompt | llm | StrOutputParser()
    
    # We chunk the text if it's too long, or pass it directly
    # For simplicity, we take the first 8000 characters to avoid context overflow
    safe_text = raw_text[:8000] 
    
    return chain.invoke({"raw_text": safe_text})