import requests
from config import console, TAVILY_API_KEY, GEMINI_API_KEY

def optimize_query(query: str) -> str:
    console.print("\n[step]▶ PHASE 0: QUERY OPTIMIZATION[/step]")
    url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash:generateContent?key={GEMINI_API_KEY}"
    prompt = f"Correct any spelling mistakes or typos in this search query. Return ONLY the corrected query string, nothing else. If it's correct, return it as is. Original: '{query}'"
    
    try:
        res = requests.post(url, json={"contents": [{"parts": [{"text": prompt}]}]})
        corrected = res.json()["candidates"][0]["content"]["parts"][0]["text"].strip()
        if corrected.lower() != query.lower():
            console.print(f"[highlight]✨ Typo detected! Auto-corrected to:[/highlight] '{corrected}'")
            return corrected
        return query
    except Exception:
        return query

def fetch_articles(query: str, max_results: int = 80) -> list:
    console.print(f"\n[step]▶ PHASE 1: SEARCHING[/step]")
    console.print(f"[info]🔍 Tavily API se '{query}' ke liye {max_results} articles fetch kar rahe hain...[/info]")
    
    url = "https://api.tavily.com/search"
    payload = {"api_key": TAVILY_API_KEY, "query": query, "search_depth": "advanced", "max_results": max_results}
    
    try:
        response = requests.post(url, json=payload)
        response.raise_for_status()
        results = response.json().get("results", [])
        console.print(f"[success]✅ Fetched {len(results)} raw articles.[/success]")
        return results
    except Exception as e:
        console.print(f"[error]❌ Tavily Search Error: {e}[/error]")
        return []
