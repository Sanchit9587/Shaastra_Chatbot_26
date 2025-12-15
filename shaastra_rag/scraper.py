import time
import sys
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.chrome.service import Service
from webdriver_manager.chrome import ChromeDriverManager
from bs4 import BeautifulSoup

def scrape_dynamic_website(url):
    """
    Scrapes a JavaScript-heavy website using Selenium with scrolling.
    """
    print(f"🌍 Starting to scrape: {url}")
    
    # Configure Headless Chrome
    chrome_options = Options()
    chrome_options.add_argument("--headless")
    chrome_options.add_argument("--no-sandbox")
    chrome_options.add_argument("--disable-dev-shm-usage")
    # Mask as a real browser to avoid being blocked
    chrome_options.add_argument("user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36")
    
    driver = None
    try:
        service = Service(ChromeDriverManager().install())
        driver = webdriver.Chrome(service=service, options=chrome_options)
        
        driver.get(url)
        print("⏳ Waiting for page to load...")
        time.sleep(3) # Initial load
        
        # --- SCROLLING LOGIC ---
        # Many sites only load content when you scroll down.
        print("📜 Scrolling to trigger content loading...")
        last_height = driver.execute_script("return document.body.scrollHeight")
        
        for _ in range(3): # Scroll down 3 times
            driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
            time.sleep(2) # Wait for new content to load
            new_height = driver.execute_script("return document.body.scrollHeight")
            if new_height == last_height:
                break
            last_height = new_height
        # -----------------------

        html_content = driver.page_source
        
        # Parse with BeautifulSoup
        soup = BeautifulSoup(html_content, 'html.parser')
        
        # Remove scripts and styles
        for script in soup(["script", "style", "svg", "path"]):
            script.extract()

        # Extract text from meaningful tags
        text_elements = soup.find_all(['p', 'h1', 'h2', 'h3', 'h4', 'li', 'span', 'div', 'article', 'section'])
        
        cleaned_texts = []
        for element in text_elements:
            text = element.get_text(separator=' ', strip=True)
            # Filter noise: Keep text that is long enough to be content
            if len(text) > 40: 
                cleaned_texts.append(text)
        
        # Dedup preserving order
        seen = set()
        unique_text = []
        for text in cleaned_texts:
            if text not in seen:
                unique_text.append(text)
                seen.add(text)
                
        final_text = "\n\n".join(unique_text)
        
        if len(final_text) < 100:
            print("⚠️ Warning: Scraped content is very short. The site might be image-based.")
        else:
            print(f"✅ Scraping complete. Extracted {len(final_text)} characters.")
            
        return final_text
        
    except Exception as e:
        print(f"❌ Scraping Error: {e}")
        return None
    finally:
        if driver:
            driver.quit()