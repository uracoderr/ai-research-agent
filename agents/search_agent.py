import requests
from config import console, logger, TAVILY_API_KEY, NVIDIA_API_KEY, MODEL_LIGHT

def optimize_query(query: str) -> str:
    console.print("\n[step]▶ PHASE 0: QUERY OPTIMIZATION[/step]")
    prompt = f"You are a strict query optimizer. Correct typos. Return ONLY the exact corrected query string. Original: '{query}'"
    
    url = "https://integrate.api.nvidia.com/v1/chat/completions"
    headers = {"Authorization": f"Bearer {NVIDIA_API_KEY}", "Content-Type": "application/json"}
    payload = {"model": MODEL_LIGHT, "messages": [{"role": "user", "content": prompt}], "temperature": 0.1, "max_tokens": 50}
    
    try:
        res = requests.post(url, json=payload, headers=headers, timeout=15)
        res.raise_for_status()
        corrected = res.json()["choices"][0]["message"]["content"].strip().replace("'", "").replace('"', "")
        if corrected.lower() != query.lower():
            console.print(f"[highlight]✨ Auto-corrected to:[/highlight] '{corrected}'")
            return corrected
        return query
    except Exception as e:
        logger.error(f"Query optimization failed: {e}")
        return query

def fetch_articles(query: str, max_results: int = 20) -> list:
    console.print(f"\n[step]▶ PHASE 1: SEARCHING WEB[/step]")
    url = "https://api.tavily.com/search"
    payload = {"api_key": TAVILY_API_KEY, "query": query, "search_depth": "advanced", "max_results": max_results}
    
    try:
        response = requests.post(url, json=payload, timeout=20)
        response.raise_for_status()
        return response.json().get("results", [])
    except Exception as e:
        logger.error(f"Tavily Search Error: {e}")
        return []
