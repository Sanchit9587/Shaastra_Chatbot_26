from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

def clean_scraped_data(raw_text, llm):
    """
    Uses the loaded LLM to summarize and clean web data.
    """
    print("🧠 Refining text with LLM...")
    
    prompt = PromptTemplate(
        template="""<|begin_of_text|><|start_header_id|>system<|end_header_id|>
You are an expert data processor.
1. Read the raw text below scraped from a website.
2. It contains noise, menus, and repeated text.
3. Extract ONLY the useful facts about Shaastra 2025 (Events, Dates, Venues, Rules).
4. Format it as clean Markdown.

RAW TEXT:
{raw_text}
<|eot_id|><|start_header_id|>assistant<|end_header_id|>
Cleaned Summary:""",
        input_variables=["raw_text"]
    )
    
    chain = prompt | llm | StrOutputParser()
    
    # Chunking: Taking top 6000 chars to fit in context. 
    # (Real scraping pipelines usually split this, but this is safe for a single page).
    safe_text = raw_text[:6000] 
    
    return chain.invoke({"raw_text": safe_text})