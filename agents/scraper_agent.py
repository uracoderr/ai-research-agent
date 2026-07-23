import requests
from bs4 import BeautifulSoup
from concurrent.futures import ThreadPoolExecutor, as_completed
from config import console, logger

def fetch_single_article(article):
    headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"}
    try:
        # ⚡ SPEED FIX: Reduced timeout to fail-fast
        res = requests.get(article["url"], headers=headers, timeout=8)
        if res.status_code == 200:
            soup = BeautifulSoup(res.text, 'html.parser')
            for script in soup(["script", "style", "nav", "footer", "header", "aside"]): script.extract()
            text = " ".join([p.get_text(strip=True) for p in soup.find_all(['p', 'h1', 'h2', 'h3', 'li'])])
            
            if len(text) > 200:
                return {"success": True, "reason": "Success", "content": f"\n\n--- SOURCE: {article.get('source_name')} | SCORE: {article.get('credibility_score')}/10 | URL: {article['url']} ---\n{text[:15000]}\n"}
            return {"success": False, "reason": "No_Content"}
        return {"success": False, "reason": f"HTTP_{res.status_code}"}
    except Exception as e: 
        return {"success": False, "reason": "Timeout/Error"}

def scrape_top_articles(ranked_articles: list, min_required: int = 10) -> tuple:
    console.print(f"\n[step]▶ PHASE 3: ASYNC SCRAPING (Target: {min_required})[/step]")
    scraped_content = ""
    stats = {"Success": 0}
    
    # ⚡ SPEED FIX: Dispatch all at once using higher workers
    batch_size = min(20, len(ranked_articles))
    
    with ThreadPoolExecutor(max_workers=batch_size) as executor:
        futures = {executor.submit(fetch_single_article, a): a for a in ranked_articles[:batch_size]}
        for future in as_completed(futures):
            res = future.result()
            if res["success"]:
                scraped_content += res["content"]
                stats["Success"] += 1
                if stats["Success"] >= min_required: break # Stop if target met

    # 🟠 HIGH FIX: Calculate REAL average credibility instead of hardcoded 8.5
    real_avg_cred = 0.0
    if ranked_articles:
        real_avg_cred = sum(a.get('credibility_score', 5) for a in ranked_articles) / len(ranked_articles)
        
    return scraped_content, stats["Success"], round(real_avg_cred, 1)
