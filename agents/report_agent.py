import requests
import json
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from config import console, NVIDIA_API_KEY

def call_nvidia_api(prompt: str, max_tokens: int = 4000, temp: float = 0.4):
    url = "https://integrate.api.nvidia.com/v1/chat/completions".strip("() '\"")
    headers = {"Authorization": f"Bearer {NVIDIA_API_KEY}", "Content-Type": "application/json"}
    payload = {
        "model": "meta/llama-3.1-70b-instruct",
        "messages": [
            {"role": "system", "content": "You are a university professor creating an academic assignment. Write with academic rigor and structure."},
            {"role": "user", "content": prompt}
        ],
        "temperature": temp,
        "max_tokens": max_tokens
    }
    
    # Simple retry logic for stability
    for _ in range(2):
        try:
            response = requests.post(url, json=payload, headers=headers, timeout=120)
            if response.status_code == 429:
                time.sleep(2)
                continue
            response.raise_for_status()
            return response.json()["choices"][0]["message"]["content"]
        except Exception as e:
            return f"Error generating section: {str(e)}"

# 🚀 SPEED FIX: Added delay parameter to stagger API calls and avoid 429 Timeouts
def generate_section(title, prompt_desc, topic, scraped_text, delay):
    time.sleep(delay)
    prompt = f"""
    Write an exhaustive academic assignment section titled '{title}' for the topic: '{topic}'.
    Instructions: {prompt_desc}
    Use the following scraped data and include inline citations based on the context:
    {scraped_text[:12000]}
    """
    content = call_nvidia_api(prompt, max_tokens=2500, temp=0.3)
    return title, content

def generate_report(topic: str, scraped_text: str, language: str, stats: dict) -> tuple:
    console.print(f"\n[step]▶ PHASE 4: FAST ASSIGNMENT GENERATION ({language.upper()})[/step]")
    
    confidence_score = max(60, min(98, int((stats['avg_credibility'] / 10) * 80)))

    lang_instruction = "Write in professional, exhaustive academic English." if "english" in language.lower() else f"Write in {language} mixing terminology appropriately."

    # 🚀 SPEED FIX: Staggered timings (0s, 1.5s, 3s, 4.5s) to perfectly bypass rate limits
    sections_config = [
        ("🎓 1. Introduction & Background", f"Provide an academic introduction, thesis statement, and define key terms. {lang_instruction}", 0),
        ("🔬 2. Literature Review & Deep Analysis", f"Critically analyze the provided data. Break down technicalities and methodologies. {lang_instruction}", 1.5),
        ("📊 3. Practical Implications & Future Outlook", f"Discuss real-world applications, challenges, and future predictions. {lang_instruction}", 3),
        ("📚 4. Conclusion & APA References", f"Summarize key findings and explicitly list the sources from the text in APA format. {lang_instruction}", 4.5)
    ]

    results = {}
    with ThreadPoolExecutor(max_workers=4) as executor:
        futures = {executor.submit(generate_section, title, desc, topic, scraped_text, delay): title for title, desc, delay in sections_config}
        for future in as_completed(futures):
            title, content = future.result()
            results[title] = content

    report_text = f"# University Assignment: {topic.title()}\n\n"
    report_text += f"> **Data Integrity Score:** {stats['avg_credibility']}/10 (Derived from {stats['scraped_success']} verified sources)\n\n"

    for title, _, _ in sections_config:
        report_text += f"\n## {title}\n"
        report_text += results.get(title, "Section generation failed.") + "\n"

    return report_text, "NVIDIA Llama-3.1 70B (Fast Staggered)"

# --- INTERACTIVE FEATURES ---
def generate_podcast_script(report_text: str):
    prompt = f"Convert this report into an engaging 2-person podcast script as a JSON array: [{{\"speaker\": \"Host A\", \"text\": \"...\"}}]. Report: {report_text[:8000]}"
    res = call_nvidia_api(prompt, max_tokens=1500, temp=0.2)
    try:
        res = res.replace("```json", "").replace("```", "").strip()
        start, end = res.find('['), res.rfind(']')
        return json.loads(res[start:end+1])
    except: return [{"speaker": "System", "text": "Failed to parse podcast script."}]

def generate_diagram(report_text: str):
    prompt = f"Create a simple Mermaid.js flowchart ('graph TD'). Use simple IDs (A,B,C). NO brackets/quotes in node text. Return ONLY raw code. Report: {report_text[:8000]}"
    res = call_nvidia_api(prompt, max_tokens=1000, temp=0.1)
    clean_code = res.replace("```mermaid", "").replace("```", "").strip()
    return "\n".join([line for line in clean_code.split('\n') if not line.lower().startswith(('here', 'sure'))])

def rag_query(context: str, query: str):
    return call_nvidia_api(f"Answer strictly based on context: {context[:20000]}\nQuestion: {query}", max_tokens=800, temp=0.1)

def challenge_query(context: str, query: str):
    return call_nvidia_api(f"Defend or critique based on context: {context[:20000]}\nChallenge: {query}", max_tokens=1200, temp=0.3)
