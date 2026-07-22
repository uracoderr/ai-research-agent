import requests
import json
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from config import console, NVIDIA_API_KEY

def call_nvidia_api(prompt: str, max_tokens: int = 4000, temp: float = 0.4):
    url = "https://integrate.api.nvidia.com/v1/chat/completions".strip("() '\"")
    headers = {"Authorization": f"Bearer {NVIDIA_API_KEY}", "Content-Type": "application/json"}
    payload = {
        "model": "meta/llama-3.1-70b-instruct",
        "messages": [
            {"role": "system", "content": "You are a senior academic research professor. Write extremely detailed, deep, and exhaustive content."},
            {"role": "user", "content": prompt}
        ],
        "temperature": temp,
        "max_tokens": max_tokens
    }
    try:
        response = requests.post(url, json=payload, headers=headers, timeout=300)
        response.raise_for_status()
        return response.json()["choices"][0]["message"]["content"]
    except Exception as e:
        return f"Error generating section: {str(e)}"

def generate_section(title, prompt_desc, topic, scraped_text):
    prompt = f"""
    Write an exhaustive, deeply detailed research section titled '{title}' for the topic: '{topic}'.
    Instructions: {prompt_desc}
    Use the following scraped data thoroughly with specific facts and details:
    {scraped_text[:15000]}
    """
    content = call_nvidia_api(prompt, max_tokens=3000, temp=0.4)
    return title, content

def generate_report(topic: str, scraped_text: str, language: str, stats: dict) -> tuple:
    console.print(f"\n[step]▶ PHASE 4: PARALLEL EXHAUSTIVE REPORT GENERATION ({language.upper()})[/step]")
    
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
        ("📑 Executive Summary & Macro Landscape", f"Provide a lengthy, highly detailed breakdown of the core landscape and macro shifts. {lang_instruction}"),
        ("🔬 Comprehensive Technical & Market Architecture", f"Provide an exhaustive deep-dive analysis breaking down core pillars, technical innovations, and market drivers. {lang_instruction}"),
        ("📈 Granular Opportunity, Challenge & Risk Matrix", f"Provide an extensive breakdown contrasting high-growth commercial opportunities against systemic risks. {lang_instruction}"),
        ("🚀 Long-term Strategic Predictions & 5-Year Roadmap", f"Provide detailed chronological timelines, predictive modeling, and strategic advice. {lang_instruction}")
    ]

    print("--> [INFO] Launching Parallel Section Generators via Llama-3.1 70B...", flush=True)
    
    results = {}
    with ThreadPoolExecutor(max_workers=4) as executor:
        futures = {executor.submit(generate_section, title, desc, topic, scraped_text): title for title, desc in sections_config}
        for future in as_completed(futures):
            title, content = future.result()
            results[title] = content
            print(f"--> [SUCCESS] Section '{title}' completed!", flush=True)

    # Compile Final Report
    report_text = f"# {topic.title()} - Comprehensive Industry Intelligence Report 2026\n\n"
    report_text += "## 📊 Executive Dashboard\n"
    report_text += f"- **Confidence Score:** {confidence_score}%\n"
    report_text += f"- **Sources Processed:** {stats['scraped_success']}\n"
    report_text += f"- **Avg Credibility:** {stats['avg_credibility']}/10\n"
    report_text += "- **Synthesis Model:** NVIDIA Llama-3.1 70B (Parallel Engine)\n\n"

    for title, _ in sections_config:
        report_text += f"\n## {title}\n"
        report_text += results.get(title, "Section generation failed.") + "\n"

    report_text += "\n## 📚 Weighted Sources & Credibility Breakdown\n"
    report_text += f"This report synthesized data from {stats['scraped_success']} verified high-credibility web sources with an average score of {stats['avg_credibility']}/10.\n"

    return report_text, "NVIDIA Llama-3.1 70B (Parallel)"

# --- INTERACTIVE FEATURES ---
def generate_podcast_script(report_text: str):
    prompt = f"Convert this report into an engaging 2-person podcast script (Host A and Expert B) as a JSON array: [{{\"speaker\": \"Host A\", \"text\": \"...\"}}]. Report: {report_text[:8000]}"
    res = call_nvidia_api(prompt, max_tokens=2000, temp=0.2)
    try:
        res = res.replace("```json", "").replace("```", "").strip()
        start, end = res.find('['), res.rfind(']')
        return json.loads(res[start:end+1])
    except:
        return [{"speaker": "System", "text": "Failed to parse podcast script."}]

def generate_diagram(report_text: str):
    prompt = f"Create a detailed Mermaid.js flowchart ('graph TD') summarizing core pillars of this report. Return ONLY valid Mermaid code without markdown blocks. Report: {report_text[:8000]}"
    return call_nvidia_api(prompt, max_tokens=1000, temp=0.1).replace("```mermaid", "").replace("```", "").strip()

def rag_query(context: str, query: str):
    prompt = f"Answer based strictly on context: {context[:30000]}\nQuestion: {query}"
    return call_nvidia_api(prompt, max_tokens=1000, temp=0.2)

def challenge_query(context: str, query: str):
    prompt = f"Defend or critique based on context: {context[:30000]}\nChallenge: {query}"
    return call_nvidia_api(prompt, max_tokens=1500, temp=0.3)
