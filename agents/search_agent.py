import requests
from config import console, TAVILY_API_KEY, NVIDIA_API_KEY

def optimize_query(query: str) -> str:
    console.print("\n[step]▶ PHASE 0: QUERY OPTIMIZATION (NVIDIA LLM)[/step]")
    prompt = f"You are a strict query optimizer. Correct any spelling mistakes or typos in this search query. Return ONLY the exact corrected query string, absolutely nothing else. Do not add quotes, intro text, or markdown. Original: '{query}'"
    
    # Clean URL explicitly
    url = "https://integrate.api.nvidia.com/v1/chat/completions".strip("() '\"")
    headers = {"Authorization": f"Bearer {NVIDIA_API_KEY}", "Content-Type": "application/json"}
    payload = {
        "model": "meta/llama-3.1-70b-instruct",
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.1,
        "max_tokens": 50
    }
    
    try:
        res = requests.post(url, json=payload, headers=headers, timeout=15)
        res.raise_for_status()
        corrected = res.json()["choices"][0]["message"]["content"].strip()
        
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
    
    # Clean URL explicitly
    url = "https://api.tavily.com/search".strip("() '\"")
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
