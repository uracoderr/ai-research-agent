import requests
import json
import time
from config import console, NVIDIA_API_KEY

def extract_json_safely(text: str):
    """LLM ke output me se strict JSON array extract karta hai chahe extra text ho"""
    try:
        start = text.find('[')
        end = text.rfind(']')
        
        if start != -1 and end != -1 and end > start:
            json_str = text[start:end+1]
            return json.loads(json_str)
        return None
    except Exception as e:
        console.print(f"[error]JSON Parse Error: {e}[/error]")
        return None

def fallback_ranker(articles, top_n):
    console.print("[warning]⚠ Fallback Ranker activated. Ranking based on standard domain patterns.[/warning]")
    for a in articles:
        a['credibility_score'] = 8 if any(d in a['url'] for d in ['.gov', '.edu', 'reuters.com', 'bloomberg.com', 'mckinsey.com', 'nature.com', 'forbes.com']) else 5
        a['source_name'] = a['url'].split('/')[2].replace('www.', '')
    return sorted(articles, key=lambda x: x['credibility_score'], reverse=True)[:top_n], 0, 0, False

def filter_and_rank_articles(articles: list, top_n: int = 20) -> tuple:
    console.print(f"\n[step]▶ PHASE 2: CREDIBILITY RANKING (NVIDIA Llama-3.1-8B)[/step]")
    
    social_domains = ['facebook.com', 'instagram.com', 'twitter.com', 'x.com', 'tiktok.com', 'youtube.com', 'reddit.com', 'pinterest.com']
    clean_articles = [a for a in articles if not any(d in a.get("url", "").lower() for d in social_domains)]
    social_dropped = len(articles) - len(clean_articles)
    
    console.print(f"[info]🗑 Pre-filtered {social_dropped} social/spam links. {len(clean_articles)} valid URLs in queue.[/info]")
    
    input_data = [{"i": i, "u": a.get("url"), "t": a.get("title", "")[:80]} for i, a in enumerate(clean_articles)]
    
    prompt = f"""
    Evaluate these sources for a deep academic/industry research report. 
    Return ONLY a raw JSON array. No markdown, no intro text.
    Format EXACTLY like this: [{{"i": 0, "s": "Reuters", "c": 9}}]
    'i' = id, 's' = short source name, 'c' = credibility score (1-10).
    Data: {json.dumps(input_data)}
    """

    api_url = "https://integrate.api.nvidia.com/v1/chat/completions".strip("() '\"")
    headers = {
        "Authorization": f"Bearer {NVIDIA_API_KEY}".strip(),
        "Content-Type": "application/json"
    }
    
    payload = {
        "model": "meta/llama-3.1-8b-instruct", # Fast 8B model for quick JSON ranking
        "messages": [
            {"role": "system", "content": "You are a JSON-only data parsing API. Output strictly valid JSON arrays without markdown blocks."},
            {"role": "user", "content": prompt}
        ], 
        "temperature": 0.1,
        "max_tokens": 3000
    }

    max_retries = 3
    for attempt in range(1, max_retries + 1):
        try:
            console.print(f"[info]🧠 LLM Ranking (Attempt {attempt}/{max_retries})...[/info]")
            
            response = requests.post(api_url, headers=headers, json=payload, timeout=60)
            response.raise_for_status()
            reply = response.json()["choices"][0]["message"]["content"]
            
            ranked_data = extract_json_safely(reply)
            if not ranked_data or not isinstance(ranked_data, list):
                raise ValueError("Extracted data is not a valid JSON list.")
            
            ranked_articles = []
            for item in ranked_data:
                idx = item.get("i")
                if idx is not None and 0 <= idx < len(clean_articles):
                    original = clean_articles[idx]
                    original['credibility_score'] = item.get("c", 6)
                    original['source_name'] = item.get("s", "Web Source")
                    ranked_articles.append(original)
            
            if not ranked_articles: 
                raise ValueError("Parsed JSON but mapping failed.")
            
            ranked_articles = sorted(ranked_articles, key=lambda x: x['credibility_score'], reverse=True)
            
            duplicates_dropped = len(clean_articles) - len(ranked_articles)
            total_spams_and_dupes = social_dropped + duplicates_dropped
            
            console.print(f"[success]✅ LLM Ranking Success! Top {len(ranked_articles)} selected.[/success]")
            return ranked_articles[:top_n], total_spams_and_dupes, attempt, True

        except Exception as e:
            console.print(f"[warning]⚠ Attempt {attempt} failed: {e}[/warning]")
            time.sleep(2)
            
    fallback_articles, _, _, _ = fallback_ranker(clean_articles, top_n)
    return fallback_articles, social_dropped, 3, False
