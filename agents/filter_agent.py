import requests
import json
from urllib.parse import urlparse
from config import console, logger, NVIDIA_API_KEY, MODEL_LIGHT

def extract_json_safely(text: str):
    try:
        start, end = text.find('['), text.rfind(']')
        if start != -1 and end != -1 and end > start:
            return json.loads(text[start:end+1])
        return None
    except Exception as e:
        logger.error(f"JSON Parsing Error: {e}")
        return None

def fallback_ranker(articles, top_n):
    for a in articles:
        a['credibility_score'] = 8 if any(d in a.get('url','') for d in ['.gov', '.edu', 'reuters', 'bloomberg']) else 5
        try:
            # 🟠 HIGH FIX: Safe URL parsing to prevent IndexError crash
            a['source_name'] = urlparse(a.get('url','')).netloc.replace('www.', '')
        except:
            a['source_name'] = "Web Source"
            
    return sorted(articles, key=lambda x: x['credibility_score'], reverse=True)[:top_n], 0, 0, False

def filter_and_rank_articles(articles: list, top_n: int = 20) -> tuple:
    console.print(f"\n[step]▶ PHASE 2: CREDIBILITY RANKING[/step]")
    social_domains = ['facebook.com', 'instagram.com', 'twitter.com', 'x.com', 'youtube.com', 'reddit.com']
    clean_articles = [a for a in articles if not any(d in a.get("url", "").lower() for d in social_domains)]
    social_dropped = len(articles) - len(clean_articles)
    
    input_data = [{"i": i, "u": a.get("url"), "t": a.get("title", "")[:80]} for i, a in enumerate(clean_articles)]
    prompt = f'Return ONLY a raw JSON array format: [{{"i": 0, "s": "Source", "c": 9}}]. Data: {json.dumps(input_data)}'

    url = "https://integrate.api.nvidia.com/v1/chat/completions"
    headers = {"Authorization": f"Bearer {NVIDIA_API_KEY}", "Content-Type": "application/json"}
    payload = {"model": MODEL_LIGHT, "messages": [{"role": "user", "content": prompt}], "temperature": 0.1, "max_tokens": 2000}

    try:
        res = requests.post(url, json=payload, headers=headers, timeout=30)
        res.raise_for_status()
        ranked_data = extract_json_safely(res.json()["choices"][0]["message"]["content"])
        
        if not ranked_data: raise ValueError("Empty JSON returned from LLM")
            
        ranked_articles = []
        for item in ranked_data:
            idx = item.get("i")
            if idx is not None and 0 <= idx < len(clean_articles):
                original = clean_articles[idx]
                original['credibility_score'] = item.get("c", 6)
                original['source_name'] = item.get("s", "Web Source")
                ranked_articles.append(original)
        
        return sorted(ranked_articles, key=lambda x: x['credibility_score'], reverse=True)[:top_n], social_dropped, 1, True
    except Exception as e:
        logger.warning(f"LLM Ranking failed, using fallback: {e}")
        return fallback_ranker(clean_articles, top_n)
