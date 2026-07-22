import requests
import json
import time
from config import console, NVIDIA_API_KEY

def extract_json_safely(text: str):
    try:
        start, end = text.find('['), text.rfind(']')
        if start != -1 and end != -1 and end > start:
            return json.loads(text[start:end+1])
        return None
    except:
        return None

def fallback_ranker(articles, top_n):
    for a in articles:
        a['credibility_score'] = 8 if any(d in a['url'] for d in ['.gov', '.edu', 'reuters.com', 'bloomberg.com', 'mckinsey.com']) else 5
        a['source_name'] = a['url'].split('/')[2].replace('www.', '')
    return sorted(articles, key=lambda x: x['credibility_score'], reverse=True)[:top_n], 0, 0, False

def filter_and_rank_articles(articles: list, top_n: int = 20) -> tuple:
    console.print(f"\n[step]▶ PHASE 2: CREDIBILITY RANKING (Fast 8B Model)[/step]")
    social_domains = ['facebook.com', 'instagram.com', 'twitter.com', 'x.com', 'youtube.com', 'reddit.com']
    clean_articles = [a for a in articles if not any(d in a.get("url", "").lower() for d in social_domains)]
    social_dropped = len(articles) - len(clean_articles)
    
    input_data = [{"i": i, "u": a.get("url"), "t": a.get("title", "")[:80]} for i, a in enumerate(clean_articles)]
    prompt = f'Return ONLY a raw JSON array format: [{{"i": 0, "s": "Source", "c": 9}}]. Data: {json.dumps(input_data)}'


    url = "https://integrate.api.nvidia.com/v1/chat/completions".strip("() '\"")
    headers = {"Authorization": f"Bearer {NVIDIA_API_KEY}", "Content-Type": "application/json"}
    payload = {
        "model": "meta/llama-3.1-8b-instruct",
        "messages": [{"role": "system", "content": "Output strictly valid JSON arrays."}, {"role": "user", "content": prompt}],
        "temperature": 0.1, "max_tokens": 2000
    }

    try:
        res = requests.post(url, json=payload, headers=headers, timeout=30)
        res.raise_for_status()
        ranked_data = extract_json_safely(res.json()["choices"][0]["message"]["content"])
        
        ranked_articles = []
        for item in ranked_data:
            idx = item.get("i")
            if idx is not None and 0 <= idx < len(clean_articles):
                original = clean_articles[idx]
                original['credibility_score'] = item.get("c", 6)
                original['source_name'] = item.get("s", "Web Source")
                ranked_articles.append(original)
        
        ranked_articles = sorted(ranked_articles, key=lambda x: x['credibility_score'], reverse=True)
        return ranked_articles[:top_n], social_dropped, 1, True
    except:
        return fallback_ranker(clean_articles, top_n)
