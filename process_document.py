# process_document.py
import fitz  # PyMuPDF
import os
import google.generativeai as genai
from tqdm import tqdm # For a progress bar

# --- CONFIGURATION ---
# Replace with your actual Gemini API Key
# For better security, use environment variables: os.getenv("GEMINI_API_KEY")
API_KEY = "AIzaSyD8yynicRgAoiuPLLc1BgQK-Q6yGbd2kO8" 
PDF_FILE_PATH = "ShaastraContextDoc.pdf"
OUTPUT_FILE_PATH = "rag_context.md"
# ---------------------

try:
    genai.configure(api_key=API_KEY)
except Exception as e:
    print(f"Error configuring Gemini API: {e}")
    print("Please ensure you have set a valid API_KEY.")
    exit()

def extract_text_from_pdf(pdf_path):
    """Extracts text from each page of a PDF."""
    if not os.path.exists(pdf_path):
        print(f"Error: PDF file not found at '{pdf_path}'")
        return None
    
    doc = fitz.open(pdf_path)
    pages_data = []
    print(f"Extracting text from '{pdf_path}'...")
    for page_num, page in enumerate(tqdm(doc, desc="Reading Pages")):
        pages_data.append({
            "page": page_num + 1,
            "text": page.get_text("text")
        })
    doc.close()
    return pages_data

def format_page_with_gemini(page_text, page_number):
    """Sends a single page's text to the Gemini API for formatting into Markdown."""
    model = genai.GenerativeModel('gemini-2.5-flash')

    prompt = f"""
    You are an expert data processor. Your task is to convert the following raw text from page {page_number} of a document into clean, well-organized Markdown format. This format is for a Retrieval-Augmented Generation (RAG) system, so clarity and structure are paramount.

    **Instructions:**
    1.  Analyze the content. If it is a schedule, format it into a Markdown table. The columns should be logical (e.g., 'Event Name', 'Start Time', 'End Time', 'Venue').
    2.  Standardize all times to a 24-hour format (e.g., '7:00-22:00' should become '19:00 - 22:00').
    3.  If the page contains rules, descriptions of events, or lists, use Markdown headings (`##`), subheadings (`###`), and bullet points (`*` or `-`).
    4.  For general paragraphs, just clean up the formatting into readable text.
    5.  Correct any obvious OCR errors you can identify from the context.
    6.  If a page is mostly empty or contains no useful text, return an empty string.
    7.  Your output must ONLY be the formatted Markdown content. Do not include any introductory phrases like "Here is the formatted Markdown:" or any other explanations.

    **Raw Text from Page {page_number}:**
    ---
    {page_text}
    ---
    """

    try:
        response = model.generate_content(prompt)
        # Clean up potential code block markers from the response
        cleaned_response = response.text.replace("```markdown", "").replace("```", "").strip()
        return cleaned_response
    except Exception as e:
        print(f"\nAn error occurred while processing page {page_number}: {e}")
        return f"### Error processing page {page_number}\n"

def process_and_save_rag_document(pages_data, output_file):
    """Iterates through pages, formats them using Gemini, and saves the final result."""
    with open(output_file, "w", encoding="utf-8") as f:
        f.write("# Shaastra 2025 - RAG Optimized Context Document\n\n")
        
        for page in tqdm(pages_data, desc="Formatting Pages with Gemini"):
            page_num = page['page']
            page_text = page['text']
            
            # Skip blank pages to save on API calls and time
            if not page_text.strip():
                continue
            
            formatted_text = format_page_with_gemini(page_text, page_num)
            
            # Write the processed content for each page to the file
            f.write(f"--- START OF PAGE {page_num} ---\n\n")
            f.write(formatted_text)
            f.write(f"\n\n--- END OF PAGE {page_num} ---\n\n")
    
    print(f"\nSuccessfully created RAG-optimized document at '{output_file}'")

if __name__ == "__main__":
    extracted_pages = extract_text_from_pdf(PDF_FILE_PATH)
    
    if extracted_pages:
        process_and_save_rag_document(extracted_pages, OUTPUT_FILE_PATH)