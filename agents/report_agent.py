import requests
import json
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from config import console, logger, NVIDIA_API_KEY, MODEL_HEAVY, MODEL_LIGHT

def call_nvidia_api(prompt: str, max_tokens: int = 3000, temp: float = 0.4, model: str = MODEL_HEAVY, retries: int = 2):
    url = "https://integrate.api.nvidia.com/v1/chat/completions"
    headers = {"Authorization": f"Bearer {NVIDIA_API_KEY}", "Content-Type": "application/json"}
    payload = {"model": model, "messages": [{"role": "system", "content": "You are a senior academic professor grading a university assignment. Write deep, academic, properly formatted content."}, {"role": "user", "content": prompt}], "temperature": temp, "max_tokens": max_tokens}
    
    for attempt in range(retries):
        try:
            response = requests.post(url, json=payload, headers=headers, timeout=100)
            if response.status_code == 429:
                time.sleep(2)
                continue
            response.raise_for_status()
            return response.json()["choices"][0]["message"]["content"]
        except Exception as e:
            if attempt == retries - 1: return f"[Error: {str(e)}]"
            time.sleep(1)

def generate_section(title, prompt_desc, topic, scraped_text, delay):
    time.sleep(delay) # Stagger to avoid rate limits
    prompt = f"Write an exhaustive academic assignment section titled '{title}' for: '{topic}'.\nInstructions: {prompt_desc}\nUse data and cite sources inline (Author, Year):\n{scraped_text[:12000]}"
    content = call_nvidia_api(prompt, max_tokens=2500, temp=0.3, model=MODEL_HEAVY)
    return title, content

def generate_report(topic: str, scraped_text: str, language: str, stats: dict) -> tuple:
    console.print(f"\n[step]▶ PHASE 4: STAGGERED ASSIGNMENT GENERATION ({language.upper()})[/step]")
    
    # ⚡ SPEED FIX: Reduced stagger delays from (0,2,4,6) to (0,1,2,3). Total time drops heavily!
    sections_config = [
        ("🎓 1. Introduction & Core Concepts", f"Provide an academic introduction, defining key terms and scope. Write in {language}.", 0),
        ("🔬 2. Literature Review & Analysis", f"Analyze the data deeply. Compare viewpoints and methodologies. Write in {language}.", 1),
        ("📊 3. Implications & Methodology", f"Discuss practical applications and future impacts. Write in {language}.", 2),
        ("📚 4. Conclusion & APA References", f"Provide a conclusion and extract actual sources from the text into an APA reference list. Write in {language}.", 3)
    ]

    results = {}
    with ThreadPoolExecutor(max_workers=4) as executor:
        futures = {executor.submit(generate_section, title, desc, topic, scraped_text, delay): title for title, desc, delay in sections_config}
        for future in as_completed(futures):
            title, content = future.result()
            results[title] = content

    report_text = f"# Academic Research Report: {topic.title()}\n\n"
    report_text += f"> **Data Integrity Score:** {stats['avg_credibility']}/10 (Derived from {stats['scraped_success']} verified sources)\n\n"

    for title, _, _ in sections_config:
        report_text += f"\n## {title}\n{results.get(title, 'Failed to generate.')}\n"

    return report_text, MODEL_HEAVY

# Interactive Tools
def generate_podcast_script(report_text: str):
    prompt = f"Convert this report into a 2-person podcast script JSON array: [{{\"speaker\": \"Host A\", \"text\": \"...\"}}]. Report: {report_text[:6000]}"
    res = call_nvidia_api(prompt, max_tokens=1500, temp=0.2, model=MODEL_LIGHT)
    try:
        json_match = __import__('re').search(r'\[\s*\{.*?\}\s*\]', res, __import__('re').DOTALL)
        if json_match: return json.loads(json_match.group(0))
        return [{"speaker": "System", "text": "Failed to parse."}]
    except: return [{"speaker": "System", "text": "Parsing Error."}]

def generate_diagram(report_text: str):
    prompt = f"Create a simple Mermaid.js flowchart ('graph TD'). Use simple IDs. NO brackets or quotes in node text. Return ONLY raw code. Report: {report_text[:6000]}"
    res = call_nvidia_api(prompt, max_tokens=800, temp=0.1, model=MODEL_LIGHT)
    clean_code = res.replace("```mermaid", "").replace("```", "").strip()
    return "\n".join([line for line in clean_code.split('\n') if not line.lower().startswith(('here', 'sure'))])

def rag_query(context: str, query: str):
    return call_nvidia_api(f"Answer strictly based on context: {context[:20000]}\nQ: {query}", max_tokens=800, temp=0.1, model=MODEL_LIGHT)

def challenge_query(context: str, query: str):
    return call_nvidia_api(f"Debate based on context: {context[:20000]}\nChallenge: {query}", max_tokens=1000, temp=0.3, model=MODEL_LIGHT)
