import time
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.chrome.service import Service
from webdriver_manager.chrome import ChromeDriverManager
from bs4 import BeautifulSoup

def scrape_dynamic_website(url):
    """
    Scrapes a JavaScript-heavy website using Selenium (Headless Chrome).
    """
    print(f"🌍 Starting to scrape: {url}")
    
    # Configure Headless Chrome
    chrome_options = Options()
    chrome_options.add_argument("--headless") # Run in background
    chrome_options.add_argument("--no-sandbox")
    chrome_options.add_argument("--disable-dev-shm-usage")
    
    try:
        # Initialize Driver
        service = Service(ChromeDriverManager().install())
        driver = webdriver.Chrome(service=service, options=chrome_options)
        
        # Load Page
        driver.get(url)
        print("⏳ Waiting for JavaScript to load content...")
        time.sleep(5) # Wait for React/JS to render the text
        
        # Get HTML
        html_content = driver.page_source
        driver.quit()
        
        # Parse with BeautifulSoup
        soup = BeautifulSoup(html_content, 'html.parser')
        
        # Extract text from meaningful tags
        text_elements = soup.find_all(['p', 'h1', 'h2', 'h3', 'li', 'span', 'div', 'article'])
        
        cleaned_texts = []
        for element in text_elements:
            text = element.get_text(separator=' ', strip=True)
            # Filter out short noise (menus, buttons)
            if len(text) > 50: 
                cleaned_texts.append(text)
        
        # Dedup
        seen = set()
        unique_text = []
        for text in cleaned_texts:
            if text not in seen:
                unique_text.append(text)
                seen.add(text)
                
        final_text = "\n\n".join(unique_text)
        
        if len(final_text) < 100:
            print("⚠️ Warning: Still scraped very little text.")
        else:
            print(f"✅ Scraping complete. Extracted {len(final_text)} characters.")
            
        return final_text
        
    except Exception as e:
        print(f"❌ Scraping Error: {e}")
        return None