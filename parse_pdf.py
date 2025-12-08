# parse_pdf.py
import nest_asyncio
nest_asyncio.apply()
from llama_parse import LlamaParse
import os

# --- CONFIGURATION ---
# Replace with your actual LlamaCloud API Key
LLAMA_CLOUD_API_KEY = "llx-QqKjasqhU89W0jFGsUys3lTfQQ2XmEnYcHsBVG9yLjxvNVTt" 
PDF_FILE_PATH = "ShaastraContextDoc.pdf"
OUTPUT_FILE_PATH = "rag_context.md"

def parse_pdf_to_markdown():
    print(f"🚀 Initializing LlamaParse for {PDF_FILE_PATH}...")
    
    parser = LlamaParse(
        api_key=LLAMA_CLOUD_API_KEY,
        result_type="markdown",  # Forces output to be pure Markdown
        verbose=True,
        language="en",
        # This argument is crucial for parsing tables inside PDFs accurately
        premium_mode=True 
    )

    print("⏳ Uploading and Parsing (this uses the cloud API)...")
    documents = parser.load_data(PDF_FILE_PATH)
    
    # Combine all pages into one string
    full_markdown_text = "\n\n".join([doc.text for doc in documents])
    
    print(f"💾 Saving to {OUTPUT_FILE_PATH}...")
    with open(OUTPUT_FILE_PATH, "w", encoding="utf-8") as f:
        f.write(full_markdown_text)
    
    print("✅ Done! Your 'rag_context.md' now has perfect table formatting.")

if __name__ == "__main__":
    parse_pdf_to_markdown()