import requests
import json
import sys
from config import console, NVIDIA_API_KEY

def call_nvidia_api(prompt: str, max_tokens: int = 4096, temp: float = 0.3, system_prompt: str = ""):
    """Utility function for all NVIDIA Llama-3.1-70B interactions"""
    url = "https://integrate.api.nvidia.com/v1/chat/completions"
    headers = {"Authorization": f"Bearer {NVIDIA_API_KEY}", "Content-Type": "application/json"}
    
    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": prompt})
    
    payload = {
        "model": "meta/llama-3.1-70b-instruct",
        "messages": messages,
        "temperature": temp,
        "max_tokens": max_tokens
    }
    
    try:
        # Timeout set to 300 seconds (5 mins) for heavy context tasks
        response = requests.post(url, json=payload, headers=headers, timeout=300)
        response.raise_for_status()
        return response.json()["choices"][0]["message"]["content"]
    except Exception as e:
        return f"Error: {str(e)}"

def generate_report(topic: str, scraped_text: str, language: str, stats: dict) -> tuple:
    console.print(f"\n[step]▶ PHASE 4: EXTENSIVE REPORT GENERATION ({language.upper()})[/step]")
    
    base_score = (stats['avg_credibility'] / 10) * 60
    volume_score = min(30, (stats['scraped_success'] / 15) * 30)
    llm_penalty = 0 if stats['llm_ranking_success'] else 15
    confidence_score = max(50, min(98, int(base_score + volume_score - llm_penalty)))

    lang_lower = language.lower()
    if "hinglish" in lang_lower:
        lang_instruction = "🚨 WRITE THE ENTIRE REPORT STRICTLY IN CONVERSATIONAL HINGLISH (Latin alphabet). Use natural conversational Hindi mixed with English terms."
    elif "hindi" in lang_lower:
        lang_instruction = "🚨 WRITE THE ENTIRE REPORT IN PURE HINDI using Devanagari script (हिंदी लिपि)."
    else:
        lang_instruction = "Write the report in deep, professional, comprehensive English."

    prompt = f"""
    You are an elite Chief Industry Analyst writing an exhaustive research report.
    Topic: '{topic}'
    {lang_instruction}
    
    REQUIRED STRUCTURE:
    # {topic.title()} - Intelligence Report
    ## 📊 Executive Dashboard
    - **Confidence Score:** {confidence_score}%
    - **Sources Scraped:** {stats['scraped_success']}
    - **Model:** [MODEL_TAG]
    ## 📑 Executive Summary
    ## 🔬 Technical & Market Analysis
    ## 📈 Opportunity & Risk Matrix
    ## 🚀 Strategic Predictions
    ## 📚 Sources Cited

    Context: {scraped_text}
    """

    print("--> [INFO] Requesting Synthesis from NVIDIA Llama-3.1 70B...", flush=True)
    report_text = call_nvidia_api(prompt, temp=0.3)
    model_used = "NVIDIA Llama-3.1 70B"

    if "Error:" in report_text:
        print(f"--> [CRITICAL] Synthesis failed: {report_text}", file=sys.stderr, flush=True)
        report_text = f"# Error Generating Report\nSynthesis failed. Details: {report_text}"
        model_used = "Failed"
    else:
        print("--> [SUCCESS] NVIDIA Llama returned the report!", flush=True)

    return report_text.replace("[MODEL_TAG]", model_used), model_used


# --- INTERACTIVE FEATURE LOGIC (ALL LLAMA 70B NOW) ---

def generate_podcast_script(report_text: str):
    system_prompt = "You are a JSON generator. Output ONLY a valid JSON array. No markdown, no explanations."
    prompt = f"""
    Convert this research report into an engaging 2-person podcast script (Host A and Expert B).
    Keep it dynamic, insightful, and around 10-15 exchanges.
    Report: {report_text[:8000]}
    
    Format EXACTLY like this:
    [{{"speaker": "Host A", "text": "..."}}, {{"speaker": "Expert B", "text": "..."}}]
    """
    res = call_nvidia_api(prompt, max_tokens=2000, temp=0.2, system_prompt=system_prompt)
    try:
        # Strict parsing to strip markdown blocks Llama sometimes adds
        res = res.replace("```json", "").replace("```", "").strip()
        start = res.find('[')
        end = res.rfind(']')
        if start != -1 and end != -1:
            return json.loads(res[start:end+1])
        return json.loads(res)
    except:
        return [{"speaker": "System", "text": f"Failed to parse podcast script. Raw output: {res[:50]}..."}]

def generate_diagram(report_text: str):
    prompt = f"""
    Read this report and create a detailed Mermaid.js flowchart (mindmap) summarizing the core pillars.
    Use 'graph TD'. Use visually distinct styles or colors if possible.
    Report: {report_text[:10000]}
    Return ONLY valid Mermaid code, nothing else. Do not use markdown code blocks like ```mermaid.
    """
    res = call_nvidia_api(prompt, max_tokens=1000, temp=0.1)
    return res.replace("```mermaid", "").replace("```", "").strip()

def rag_query(context: str, query: str):
    prompt = f"""
    You are an AI assistant answering questions based STRICTLY on the provided context.
    If the answer isn't in the context, say "I cannot find this in the scraped data."
    Context: {context[:30000]}
    Question: {query}
    """
    return call_nvidia_api(prompt, max_tokens=1000, temp=0.2)

def challenge_query(context: str, query: str):
    prompt = f"""
    You are a rigorous academic defender. The user is challenging the report's findings.
    Evaluate their challenge using the raw scraped data provided. 
    If they are right, concede and correct the record. If they are wrong, fiercely defend the report using citations from the context.
    Context: {context[:30000]}
    Challenge: {query}
    """
    return call_nvidia_api(prompt, max_tokens=1500, temp=0.3)
