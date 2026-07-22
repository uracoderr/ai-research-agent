import requests
from config import console, TAVILY_API_KEY, GEMINI_API_KEY

def optimize_query(query: str) -> str:
    console.print("\n[step]▶ PHASE 0: QUERY OPTIMIZATION[/step]")
    url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash:generateContent?key={GEMINI_API_KEY}"
    
    # Made prompt extremely strict to avoid conversational junk
    prompt = f"You are a strict query optimizer. Correct any spelling mistakes or typos in this search query. Return ONLY the exact corrected query string, absolutely nothing else. Do not add quotes, intro text, or markdown. Original: '{query}'"
    
    try:
        res = requests.post(url, json={"contents": [{"parts": [{"text": prompt}]}]}, timeout=15)
        res.raise_for_status()
        corrected = res.json()["candidates"][0]["content"]["parts"][0]["text"].strip()
        
        # Clean up any quotes if the LLM still adds them
        corrected = corrected.replace("'", "").replace('"', "")
        
        if corrected.lower() != query.lower():
            console.print(f"[highlight]✨ Typo detected! Auto-corrected to:[/highlight] '{corrected}'")
            return corrected
        return query
    except Exception as e:
        console.print(f"[warning]⚠ Query optimization failed, using raw query. ({str(e)})[/warning]")
        return query

def fetch_articles(query: str, max_results: int = 20) -> list:
    console.print(f"\n[step]▶ PHASE 1: SEARCHING WEB[/step]")
    console.print(f"[info]🔍 Tavily API se '{query}' ke liye {max_results} articles fetch kar rahe hain...[/info]")
    
    url = "https://api.tavily.com/search"
    payload = {
        "api_key": TAVILY_API_KEY, 
        "query": query, 
        "search_depth": "advanced", 
        "max_results": max_results
    }
    
    try:
        response = requests.post(url, json=payload, timeout=20)
        response.raise_for_status()
        results = response.json().get("results", [])
        
        if results:
            console.print(f"[success]✅ Fetched {len(results)} raw articles.[/success]")
        else:
            console.print(f"[warning]⚠ No results found for the query.[/warning]")
            
        return results
    except Exception as e:
        console.print(f"[error]❌ Tavily Search Error: {e}[/error]")
        return []
