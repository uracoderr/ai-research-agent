import requests
import json
import sys
from config import console, NVIDIA_API_KEY

def call_nvidia_api(prompt: str, max_tokens: int = 8192, temp: float = 0.4, system_prompt: str = ""):
    """Utility function for all NVIDIA Llama-3.1-70B interactions"""
    url = "https://integrate.api.nvidia.com/v1/chat/completions".strip("() '\"")
    headers = {"Authorization": f"Bearer {NVIDIA_API_KEY}", "Content-Type": "application/json"}
    
    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    else:
        messages.append({"role": "system", "content": "You are an exhaustive, verbose, and senior academic research professor. Never write short summaries. Write deeply comprehensive long-form content."})
        
    messages.append({"role": "user", "content": prompt})
    
    payload = {
        "model": "meta/llama-3.1-70b-instruct",
        "messages": messages,
        "temperature": temp,
        "max_tokens": max_tokens
    }
    
    try:
        response = requests.post(url, json=payload, headers=headers, timeout=300)
        response.raise_for_status()
        return response.json()["choices"][0]["message"]["content"]
    except Exception as e:
        return f"Error: {str(e)}"

def generate_report(topic: str, scraped_text: str, language: str, stats: dict) -> tuple:
    console.print(f"\n[step]▶ PHASE 4: GENERATING EXHAUSTIVE DEEP-DIVE REPORT ({language.upper()})[/step]")
    
    base_score = (stats['avg_credibility'] / 10) * 60
    volume_score = min(30, (stats['scraped_success'] / 15) * 30)
    llm_penalty = 0 if stats['llm_ranking_success'] else 15
    confidence_score = max(50, min(98, int(base_score + volume_score - llm_penalty)))

    lang_lower = language.lower()
    if "hinglish" in lang_lower:
        lang_instruction = "🚨 WRITE THE ENTIRE REPORT STRICTLY IN CONVERSATIONAL HINGLISH (Latin alphabet). Use natural conversational Hindi mixed with English terms extensively across every paragraph."
    elif "hindi" in lang_lower:
        lang_instruction = "🚨 WRITE THE ENTIRE REPORT IN PURE HINDI using Devanagari script (हिंदी लिपि)."
    else:
        lang_instruction = "Write the report in deep, highly professional, exhaustive academic English."

    prompt = f"""
    You are an elite Chief Industry Analyst and Lead Academic Researcher writing an massive, exhaustive, long-form intelligence report for 2026. 
    CRITICAL INSTRUCTION: Do NOT write a short summary or brief overview. Every single section below MUST be thoroughly expanded with multiple detailed paragraphs, deep technical breakdowns, specific market data points, and granular explanations. If a section is short, the report fails.
    
    Topic: '{topic}'
    {lang_instruction}
    
    Use the provided scraped data thoroughly. Cite sources appropriately using inline tags.
    
    MANDATORY DEEP-DIVE STRUCTURE (Expand each section extensively):
    
    # {topic.title()} - Comprehensive Industry Intelligence Report 2026
    
    ## 📊 Executive Dashboard
    - **Confidence Score:** {confidence_score}%
    - **Sources Processed & Scraped:** {stats['scraped_success']}
    - **Avg Source Credibility:** {stats['avg_credibility']}/10
    - **Synthesis Model:** [MODEL_TAG]
    
    ## 📑 Executive Summary & Macro Landscape
    (Write at least 3-4 detailed paragraphs explaining the overarching thesis, macro market shifts, historical context, and immediate industry takeaways.)
    
    ## 🔬 Comprehensive Technical & Market Architecture
    (Write an exhaustive deep-dive analysis consisting of multiple subsections. Break down core pillars, underlying mechanics, technical innovations, and current market adoption drivers with extreme granularity.)
    
    ## 📈 Granular Opportunity, Challenge & Risk Matrix
    (Provide an extensive, multi-paragraph breakdown contrasting high-growth commercial opportunities against systemic technical, regulatory, and financial risks.)
    
    ## 🚀 Long-term Strategic Predictions & 5-Year Actionable Roadmap
    (Provide detailed chronological timelines (Year 1 to Year 5), predictive modeling, and strategic advice for stakeholders and developers.)
    
    ## 📚 Weighted Sources & Source Credibility Breakdown
    (Exhaustively list and evaluate the key sources utilized in this synthesis, detailing their reliability and contribution.)

    Scraped Data Context:
    {scraped_text}
    """

    print("--> [INFO] Requesting Massive Exhaustive Synthesis from NVIDIA Llama-3.1 70B...", flush=True)
    report_text = call_nvidia_api(prompt, max_tokens=8192, temp=0.4)
    model_used = "NVIDIA Llama-3.1-70B"

    if "Error:" in report_text or len(report_text) < 500:
        print(f"--> [CRITICAL] Synthesis failed or too short: {report_text}", file=sys.stderr, flush=True)
        report_text = f"# Error Generating Report\nSynthesis failed or returned insufficient content. Details: {report_text}"
        model_used = "Failed"
    else:
        print(f"--> [SUCCESS] Exhaustive report generated successfully! Length: {len(report_text)} chars", flush=True)

    return report_text.replace("[MODEL_TAG]", model_used), model_used


# --- INTERACTIVE FEATURE LOGIC ---

def generate_podcast_script(report_text: str):
    system_prompt = "You are a JSON generator. Output ONLY a valid JSON array. No markdown, no explanations."
    prompt = f"""
    Convert this research report into an engaging 2-person podcast script (Host A and Expert B).
    Keep it dynamic, insightful, and around 10-15 exchanges.
    Report: {report_text[:10000]}
    
    Format EXACTLY like this:
    [{{"speaker": "Host A", "text": "..."}}, {{"speaker": "Expert B", "text": "..."}}]
    """
    res = call_nvidia_api(prompt, max_tokens=2500, temp=0.3, system_prompt=system_prompt)
    try:
        res = res.replace("```json", "").replace("```", "").strip()
        start = res.find('[')
        end = res.rfind(']')
        if start != -1 and end != -1:
            return json.loads(res[start:end+1])
        return json.loads(res)
    except:
        return [{"speaker": "System", "text": f"Failed to parse podcast script."}]

def generate_diagram(report_text: str):
    prompt = f"""
    Read this report and create a detailed Mermaid.js flowchart (mindmap) summarizing the core pillars.
    Use 'graph TD'. Use visually distinct styles or colors if possible.
    Report: {report_text[:12000]}
    Return ONLY valid Mermaid code, nothing else. Do not use markdown code blocks like ```mermaid.
    """
    res = call_nvidia_api(prompt, max_tokens=1500, temp=0.1)
    return res.replace("```mermaid", "").replace("```", "").strip()

def rag_query(context: str, query: str):
    prompt = f"""
    You are an AI research assistant answering questions based STRICTLY on the provided context.
    If the answer isn't in the context, say "I cannot find this in the scraped data."
    Context: {context[:35000]}
    Question: {query}
    """
    return call_nvidia_api(prompt, max_tokens=2000, temp=0.2)

def challenge_query(context: str, query: str):
    prompt = f"""
    You are a rigorous academic defender. The user is challenging the report's findings.
    Evaluate their challenge using the raw scraped data provided. 
    If they are right, concede and correct the record. If they are wrong, fiercely defend the report using citations from the context.
    Context: {context[:35000]}
    Challenge: {query}
    """
    return call_nvidia_api(prompt, max_tokens=2500, temp=0.3)
