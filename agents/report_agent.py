import requests
import json
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from config import console, NVIDIA_API_KEY

# 🚀 FIX 1: Added Model Selection & Retry Logic
def call_nvidia_api(prompt: str, max_tokens: int = 4000, temp: float = 0.4, model: str = "meta/llama-3.1-70b-instruct", retries: int = 2):
    url = "https://integrate.api.nvidia.com/v1/chat/completions"
    headers = {"Authorization": f"Bearer {NVIDIA_API_KEY}", "Content-Type": "application/json"}
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": "You are an elite academic researcher. Always output exactly what is asked. No introductory filler text."},
            {"role": "user", "content": prompt}
        ],
        "temperature": temp,
        "max_tokens": max_tokens
    }
    
    for attempt in range(retries):
        try:
            response = requests.post(url, json=payload, headers=headers, timeout=120)
            if response.status_code == 429: # Rate limit hit
                time.sleep(2 * (attempt + 1))
                continue
            response.raise_for_status()
            return response.json()["choices"][0]["message"]["content"]
        except Exception as e:
            if attempt == retries - 1:
                return f"[API Error or Rate Limit after {retries} attempts: {str(e)}]"
            time.sleep(2)

def generate_section(title, prompt_desc, topic, scraped_text, delay):
    time.sleep(delay) # 🚀 FIX 2: Staggering to prevent API blocking
    prompt = f"""
    Write an exhaustive, deeply detailed research section titled '{title}' for the topic: '{topic}'.
    Instructions: {prompt_desc}
    Use the following scraped data thoroughly with specific facts, numbers, and details. Do NOT summarize, expand deeply!
    Data:
    {scraped_text[:12000]}
    """
    # Using 70B for the main report for high quality
    content = call_nvidia_api(prompt, max_tokens=3500, temp=0.5, model="meta/llama-3.1-70b-instruct")
    return title, content

def generate_report(topic: str, scraped_text: str, language: str, stats: dict) -> tuple:
    console.print(f"\n[step]▶ PHASE 4: STAGGERED PARALLEL REPORT GENERATION ({language.upper()})[/step]")
    
    base_score = (stats['avg_credibility'] / 10) * 60
    volume_score = min(30, (stats['scraped_success'] / 15) * 30)
    llm_penalty = 0 if stats['llm_ranking_success'] else 15
    confidence_score = max(50, min(98, int(base_score + volume_score - llm_penalty)))

    lang_lower = language.lower()
    if "hinglish" in lang_lower:
        lang_instruction = "Write strictly in conversational Hinglish (Latin alphabet), mixing Hindi and English naturally."
    elif "hindi" in lang_lower:
        lang_instruction = "Write purely in Hindi using Devanagari script."
    else:
        lang_instruction = "Write in professional, exhaustive academic English."

    sections_config = [
        ("📊 Executive Summary & Macro Landscape", f"Provide a lengthy, highly detailed breakdown of the core landscape and macro shifts. {lang_instruction}", 0),
        ("⚙️ Comprehensive Technical & Market Architecture", f"Provide an exhaustive deep-dive analysis breaking down core pillars, technical innovations, and market drivers. {lang_instruction}", 2),
        ("⚠️ Granular Opportunity, Challenge & Risk Matrix", f"Provide an extensive breakdown contrasting high-growth commercial opportunities against systemic risks. {lang_instruction}", 4),
        ("🔮 Long-term Strategic Predictions & 5-Year Roadmap", f"Provide detailed chronological timelines, predictive modeling, and strategic advice. {lang_instruction}", 6)
    ]

    print("--> [INFO] Launching Smart Parallel Section Generators (Staggered)...", flush=True)
    
    results = {}
    with ThreadPoolExecutor(max_workers=4) as executor:
        # We pass the delay to stagger the API calls
        futures = {executor.submit(generate_section, title, desc, topic, scraped_text, delay): title for title, desc, delay in sections_config}
        for future in as_completed(futures):
            title, content = future.result()
            results[title] = content
            print(f"--> [SUCCESS] Section '{title}' completed!", flush=True)

    report_text = f"# {topic.title()} - Comprehensive Industry Intelligence Report\n\n"
    report_text += "## 🎯 Executive Dashboard\n"
    report_text += f"- **Confidence Score:** {confidence_score}%\n"
    report_text += f"- **Sources Processed:** {stats['scraped_success']}\n"
    report_text += f"- **Avg Credibility:** {stats['avg_credibility']}/10\n"
    report_text += "- **Synthesis Model:** Llama-3.1 70B (Staggered Engine)\n\n"

    # Assemble in order
    for title, _, _ in sections_config:
        report_text += f"\n## {title}\n"
        report_text += results.get(title, "Section generation failed.") + "\n"

    return report_text, "NVIDIA Llama-3.1 70B (Parallel)"

# --- 🚀 FIX 3: BULLETPROOF INTERACTIVE FEATURES (Using 8B Model for Speed) ---

def generate_podcast_script(report_text: str):
    prompt = f"Convert this report into a fun 2-person podcast script. \nRETURN STRICTLY A JSON ARRAY. No explanations.\nFormat: [{{\"speaker\": \"Host A\", \"text\": \"...\"}}]\nReport: {report_text[:8000]}"
    # Switch to 8B for fast JSON generation
    res = call_nvidia_api(prompt, max_tokens=2000, temp=0.2, model="meta/llama-3.1-8b-instruct")
    try:
        # Robust Regex to find JSON array even if LLM adds garbage text around it
        json_match = re.search(r'\[\s*\{.*?\}\s*\]', res, re.DOTALL)
        if json_match:
            return json.loads(json_match.group(0))
        return [{"speaker": "System", "text": "Sorry, failed to generate script properly."}]
    except:
        return [{"speaker": "System", "text": "JSON Parsing Error."}]

def generate_diagram(report_text: str):
    prompt = f"""
    You are a strict Mermaid.js compiler. Create a flowchart summarizing the core pillars of the report.
    CRITICAL RULES:
    1. Start exactly with 'graph TD'.
    2. Use simple letters for Node IDs (e.g., A, B, C).
    3. Node text MUST NOT contain parentheses (), square brackets [], quotes "", or special characters. Use plain letters and spaces only.
    4. Example: A[Data Analysis] --> B[Market Trends]
    5. RETURN ONLY RAW MERMAID CODE. NO MARKDOWN, NO BACKTICKS.
    
    Report: {report_text[:6000]}
    """
    try:
        # Calling your existing API function
        res = call_nvidia_api(prompt, max_tokens=1000, temp=0.1)
        
        # 1. Clean markdown backticks
        res = res.replace("```mermaid", "").replace("```", "").strip()
        
        # 2. AGGRESSIVE CLEANER: Cut everything before 'graph TD' (Removes "Here is your code")
        if "graph " in res:
            res = "graph " + res.split("graph ")[1]
            
        # 3. Filter out random conversational lines LLM might add at the end
        valid_lines = []
        for line in res.split('\n'):
            line = line.strip()
            if not line.lower().startswith(('here', 'sure', 'note:', 'this diagram')):
                valid_lines.append(line)
                
        final_mermaid = "\n".join(valid_lines)
        return final_mermaid
        
    except Exception as e:
        # Fail-safe: Agar error aaye toh UI crash hone ki jagah error node dikhaye
        safe_error = str(e).replace('(', '').replace(')', '').replace('[', '').replace(']', '')
        return f"graph TD\n Z[Diagram Error] --> Y[{safe_error}]"


def rag_query(context: str, query: str):
    prompt = f"Answer this query based ONLY on the context provided.\nContext: {context[:20000]}\nQuestion: {query}"
    # 8B for fast chatting
    return call_nvidia_api(prompt, max_tokens=1000, temp=0.2, model="meta/llama-3.1-8b-instruct")

def challenge_query(context: str, query: str):
    prompt = f"Act as a critical debater. Defend or critique this claim based on the context.\nContext: {context[:20000]}\nChallenge: {query}"
    # 8B for fast chatting
    return call_nvidia_api(prompt, max_tokens=1500, temp=0.3, model="meta/llama-3.1-8b-instruct")
