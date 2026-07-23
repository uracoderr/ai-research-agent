import requests
from bs4 import BeautifulSoup
from concurrent.futures import ThreadPoolExecutor, as_completed
from config import console

def fetch_single_article(article):
    headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0"}
    try:
        # 🚀 SPEED FIX: Reduced timeout to 8s to fail-fast on slow websites
        res = requests.get(article["url"], headers=headers, timeout=8)
        if res.status_code == 200:
            soup = BeautifulSoup(res.text, 'html.parser')
            for script in soup(["script", "style", "nav", "footer", "header", "aside"]):
                script.extract()
            text = " ".join([p.get_text(strip=True) for p in soup.find_all(['p', 'h1', 'h2', 'h3', 'li'])])
            
            if len(text) > 200:
                return {
                    "success": True, 
                    "reason": "Success", 
                    "content": f"\n\n--- SOURCE: {article.get('source_name')} | SCORE: {article.get('credibility_score')}/10 | URL: {article['url']} ---\n{text[:12000]}\n"
                }
            return {"success": False, "reason": "No_Content"}
        elif res.status_code in [401, 403]: return {"success": False, "reason": "Blocked"}
        else: return {"success": False, "reason": f"HTTP_{res.status_code}"}
    except Exception: return {"success": False, "reason": "Timeout/Error"}

def scrape_top_articles(ranked_articles: list, min_required: int = 10) -> tuple:
    console.print(f"\n[step]▶ PHASE 3: ASYNC SCRAPING & DEEP CONTEXT EXTRACTION (Target: {min_required}+)[/step]")
    scraped_content = ""
    stats = {"Requested": 0, "Success": 0}
    
    # 🚀 SPEED FIX: Batch size increased to blast through the URLs quickly
    batch_size = 20
    
    for i in range(0, len(ranked_articles), batch_size):
        if stats["Success"] >= min_required: break
            
        batch = ranked_articles[i : i+batch_size]
        stats["Requested"] += len(batch)
        
        with ThreadPoolExecutor(max_workers=batch_size) as executor:
            futures = {executor.submit(fetch_single_article, a): a for a in batch}
            for future in as_completed(futures):
                res = future.result()
                if res["success"]:
                    scraped_content += res["content"]
                    stats["Success"] += 1
                    
    return scraped_content, stats["Success"]
