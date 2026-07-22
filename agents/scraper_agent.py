import requests
from bs4 import BeautifulSoup
from concurrent.futures import ThreadPoolExecutor, as_completed
from config import console

def fetch_single_article(article):
    headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0"}
    try:
        # Timeout thoda badhaya hai taaki heavy pages bhi load ho sake
        res = requests.get(article["url"], headers=headers, timeout=12)
        if res.status_code == 200:
            soup = BeautifulSoup(res.text, 'html.parser')
            
            # Junk tags remove kar rahe hain clean text ke liye
            for script in soup(["script", "style", "nav", "footer", "header", "aside"]):
                script.extract()
                
            # headings aur lists bhi include kiye hain sirf paras nahi
            text = " ".join([p.get_text(strip=True) for p in soup.find_all(['p', 'h1', 'h2', 'h3', 'li'])])
            
            if len(text) > 200:
                # 🌟 IMPORTANT FIX: Limit increased from 2000 to 15000 chars for Deep RAG Context!
                return {
                    "success": True, 
                    "reason": "Success", 
                    "content": f"\n\n--- SOURCE: {article.get('source_name')} | SCORE: {article.get('credibility_score')}/10 | URL: {article['url']} ---\n{text[:15000]}\n"
                }
            return {"success": False, "reason": "No_Content"}
        elif res.status_code in [401, 403]: return {"success": False, "reason": "Blocked_403"}
        else: return {"success": False, "reason": f"HTTP_{res.status_code}"}
    except requests.exceptions.Timeout: return {"success": False, "reason": "Timeout"}
    except Exception: return {"success": False, "reason": "Error"}

def scrape_top_articles(ranked_articles: list, min_required: int = 15) -> tuple:
    console.print(f"\n[step]▶ PHASE 3: ASYNC SCRAPING & DEEP CONTEXT EXTRACTION (Target: {min_required}+)[/step]")
    scraped_content = ""
    stats = {"Requested": 0, "Success": 0, "Blocked_403": 0, "Timeout": 0, "No_Content": 0, "Other_Errors": 0}
    batch_size = 15
    
    for i in range(0, len(ranked_articles), batch_size):
        if stats["Success"] >= min_required: break
            
        batch = ranked_articles[i : i+batch_size]
        stats["Requested"] += len(batch)
        console.print(f"[info]⚡ Dispatching Scraping Batch {i//batch_size + 1} ({len(batch)} URLs)...[/info]")
        
        with ThreadPoolExecutor(max_workers=batch_size) as executor:
            futures = {executor.submit(fetch_single_article, a): a for a in batch}
            for future in as_completed(futures):
                res = future.result()
                if res["success"]:
                    scraped_content += res["content"]
                    stats["Success"] += 1
                else:
                    reason = res["reason"]
                    stats[reason] = stats.get(reason, 0) + 1
                    
        console.print(f"[highlight]↳ Scraped so far: {stats['Success']} (Target met: {min_required}+)[/highlight]")

    console.print("\n[success]📊 --- Scraping Summary Matrix ---[/success]")
    for k, v in stats.items():
        console.print(f"  {k:<13}: {v}")
        
    return scraped_content, stats["Success"]

