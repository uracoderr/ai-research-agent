import requests
import json
import sys
import re
from config import console, GEMINI_API_KEY, NVIDIA_API_KEY

def call_gemini_api(prompt: str, json_mode: bool = False):
    """Utility function for all Gemini interactions"""
    url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash:generateContent?key={GEMINI_API_KEY}"
    payload = {"contents": [{"parts": [{"text": prompt}]}]}
    if json_mode:
        payload["generationConfig"] = {"responseMimeType": "application/json"}
        
    try:
        response = requests.post(url, json=payload, headers={"Content-Type": "application/json"}, timeout=1440)
        if response.status_code == 200:
            return response.json().get("candidates", [{}])[0].get("content", {}).get("parts", [{}])[0].get("text", "")
        return f"Error: API returned status {response.status_code}"
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

    model_used = "Gemini 2.5 Flash"
    report_text = call_gemini_api(prompt)

    if "Error:" in report_text or not report_text.strip():
        print("--> [INFO] Falling back to NVIDIA Llama-3.1 70B...", flush=True)
        nvidia_url = "https://integrate.api.nvidia.com/v1/chat/completions"
        headers = {"Authorization": f"Bearer {NVIDIA_API_KEY}", "Content-Type": "application/json"}
        nvidia_payload = {
            "model": "meta/llama-3.1-70b-instruct",
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.3, "max_tokens": 4096
        }
        try:
            res = requests.post(nvidia_url, headers=headers, json=nvidia_payload, timeout=60)
            res.raise_for_status()
            report_text = res.json()["choices"][0]["message"]["content"]
            model_used = "NVIDIA Llama-3.1 70B"
        except Exception as e:
            report_text = f"# Error\nSynthesis failed: {str(e)}"
            model_used = "Failed"

    return report_text.replace("[MODEL_TAG]", model_used), model_used


# --- NEW INTERACTIVE FEATURE LOGIC ---

def generate_podcast_script(report_text: str):
    prompt = f"""
    Convert this research report into an engaging 2-person podcast script (Host A and Expert B).
    Keep it dynamic, insightful, and around 10-15 exchanges.
    Report: {report_text[:8000]}
    
    Output strictly in this JSON format:
    [{{"speaker": "Host A", "text": "..."}}, {{"speaker": "Expert B", "text": "..."}}]
    """
    res = call_gemini_api(prompt, json_mode=True)
    try:
        # Clean any accidental markdown backticks
        res = res.replace("```json", "").replace("```", "").strip()
        return json.loads(res)
    except:
        return [{"speaker": "System", "text": "Failed to parse podcast script."}]

def generate_diagram(report_text: str):
    prompt = f"""
    Read this report and create a detailed Mermaid.js flowchart (mindmap) summarizing the core pillars.
    Use 'graph TD'. Use visually distinct styles or colors if possible.
    Report: {report_text[:10000]}
    Return ONLY valid Mermaid code, nothing else. No markdown blocks.
    """
    res = call_gemini_api(prompt)
    return res.replace("```mermaid", "").replace("```", "").strip()

def rag_query(context: str, query: str):
    prompt = f"""
    You are an AI assistant answering questions based STRICTLY on the provided context.
    If the answer isn't in the context, say "I cannot find this in the scraped data."
    Context: {context[:30000]}
    Question: {query}
    """
    return call_gemini_api(prompt)

def challenge_query(context: str, query: str):
    prompt = f"""
    You are a rigorous academic defender. The user is challenging the report's findings.
    Evaluate their challenge using the raw scraped data provided. 
    If they are right, concede and correct the record. If they are wrong, fiercely defend the report using citations from the context.
    Context: {context[:30000]}
    Challenge: {query}
    """
    return call_gemini_api(prompt)
