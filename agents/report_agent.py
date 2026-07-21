import requests
import json
from config import console, GEMINI_API_KEY, NVIDIA_API_KEY

def generate_report(topic: str, scraped_text: str, language: str, stats: dict) -> str:
    console.print(f"\n[step]▶ PHASE 4: REPORT GENERATION (GEMINI / NVIDIA FALLBACK)[/step]")
    
    # Realistic Dynamic Confidence Calculation
    base_score = (stats['avg_credibility'] / 10) * 60
    volume_score = min(30, (stats['scraped_success'] / 15) * 30)
    llm_penalty = 0 if stats['llm_ranking_success'] else 15
    confidence_score = max(50, min(98, int(base_score + volume_score - llm_penalty)))

    prompt = f"""
    You are an elite Industry Analyst writing a production-grade research report in the year 2026.
    Topic: '{topic}'
    Language strictly: {language} (If Hinglish: Use ONLY Latin/Roman alphabet, mix conversational Hindi with English tech terms. No Devanagari).
    
    Use the provided scraped data. Pay attention to the "Score" and "Source" of each snippet.
    
    REQUIRED REPORT STRUCTURE:
    
    # [Topic] - Industry Intelligence Report 2026
    
    ## 📊 Executive Dashboard
    - **Confidence Score:** {confidence_score}%
    - **Sources Processed & Scraped:** {stats['scraped_success']}
    - **Avg Source Credibility:** {stats['avg_credibility']}/10
    - **Junk/Duplicates Filtered:** {stats['duplicates_removed']}
    - **Ranking Engine Status:** {'NVIDIA Llama-3.1 (AI Active)' if stats['llm_ranking_success'] else 'Fallback Rule-Based'}
    
    ## 📑 Executive Summary & Key Takeaways
    (High level summary and bullet points of immediate takeaways)
    
    ## 🔬 Detailed Analysis, Contradictions & Data Points
    (Write deep dive here with inline citations like [Reuters] or [OpenAI] and highlight contradictions)
    
    ## 📈 Opportunity & Risk Matrix
    (Structured breakdown or table of Risks vs Opportunities)
    
    ## 🚀 Future Predictions & Action Items
    (Timeline/Predictions for next 2-5 years and actionable advice)
    
    ## 📚 Weighted Sources Used
    (List sources used with their respective credibility weights/stars)

    Data context:
    {scraped_text}
    """

    # 🌟 Try Gemini First
    gemini_url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash:generateContent?key={GEMINI_API_KEY}"
    payload = {"contents": [{"parts": [{"text": prompt}]}]}

    try:
        response = requests.post(gemini_url, json=payload, headers={"Content-Type": "application/json"}, timeout=30)
        if response.status_code == 200:
            return response.json()["candidates"][0]["content"]["parts"][0]["text"]
        else:
            console.print(f"[warning]⚠ Gemini API Quota/Error ({response.status_code}). Switching to NVIDIA Fallback...[/warning]")
    except Exception as e:
        console.print(f"[warning]⚠ Gemini Connection Failed: {e}. Switching to NVIDIA Fallback...[/warning]")

    # 🌟 Fallback to NVIDIA Llama if Gemini fails/quota full
    nvidia_url = "https://integrate.api.nvidia.com/v1/chat/completions"
    headers = {"Authorization": f"Bearer {NVIDIA_API_KEY}", "Content-Type": "application/json"}
    nvidia_payload = {
        "model": "meta/llama-3.1-8b-instruct",
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.3,
        "max_tokens": 3000
    }

    try:
        res = requests.post(nvidia_url, headers=headers, json=nvidia_payload, timeout=40)
        res.raise_for_status()
        return res.json()["choices"][0]["message"]["content"]
    except Exception as e:
        return f"# Error Generating Report\nBoth Gemini and NVIDIA LLMs failed. Details: {str(e)}"
