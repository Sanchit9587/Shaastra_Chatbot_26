import sys
import os

# --- PATH FIX ---
# Add the current directory to Python's path so it finds 'shaastra_rag'
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
# ----------------

from shaastra_rag import config
from shaastra_rag.models import load_llm
from shaastra_rag.scraper import scrape_dynamic_website
from shaastra_rag.refiner import clean_scraped_data

# URL to scrape (Shaastra Main Page)
TARGET_URL = "https://www.shaastra.org/"

def update_knowledge_base():
    # 1. Scrape
    raw_text = scrape_dynamic_website(TARGET_URL)
    
    if not raw_text or len(raw_text) < 100:
        print("❌ Scraper failed or site has no text content (Image only).")
        print("⚠️ Skipping update to avoid corrupting the knowledge base.")
        return

    # 2. Load Model
    llm = load_llm(config.LLM_MODEL_ID)
    
    # 3. Refine
    cleaned_markdown = clean_scraped_data(raw_text, llm)
    
    print("\n--- REFINED OUTPUT PREVIEW ---\n")
    print(cleaned_markdown[:500] + "...\n")
    
    # 4. Save to Context File
    output_path = config.CONTEXT_FILE
    
    with open(output_path, "a", encoding="utf-8") as f:
        f.write("\n\n# --- SCRAPED WEBSITE DATA ---\n\n")
        f.write(cleaned_markdown)
        
    print(f"✅ Successfully appended data to {output_path}")

if __name__ == "__main__":
    update_knowledge_base()