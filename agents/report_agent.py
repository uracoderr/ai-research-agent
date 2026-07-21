import requests
from config import console, GEMINI_API_KEY

def generate_report(topic: str, scraped_text: str, language: str, stats: dict) -> str:
    console.print(f"\n[step]▶ PHASE 4: REPORT GENERATION (GEMINI)[/step]")
    
    url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash:generateContent?key={GEMINI_API_KEY}"
    
    # 🌟 Realistic Dynamic Confidence Calculation
    base_score = (stats['avg_credibility'] / 10) * 60
    volume_score = min(30, (stats['scraped_success'] / 15) * 30)
    llm_penalty = 0 if stats['llm_ranking_success'] else 15 # Agar fallback hua toh 15% penalty
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
    (Write deep dive here. 
    CRITICAL RULES:
    1. INLINE CITATIONS: Use inline citations like [McKinsey] or [IBM, Reuters] within text.
    2. MERGE EVIDENCE: If multiple sources say the same thing, merge them: "Multiple sources [IBM, Forbes] indicate..."
    3. CONTRADICTION DETECTION: Explicitly highlight statistical contradictions if different sources claim opposing metrics (e.g., Market size estimates differing between reports).)
    
    ## 📈 Opportunity & Risk Matrix
    (Structured breakdown or table of Risks vs Opportunities)
    
    ## 🚀 Future Predictions & Action Items
    (Timeline/Predictions for next 2-5 years and actionable advice)
    
    ## 📚 Weighted Sources Used
    (List sources used with their respective credibility weights/stars)

    Data context:
    {scraped_text}
    """

    payload = {"contents": [{"parts": [{"text": prompt}]}]}

    try:
        response = requests.post(url, json=payload, headers={"Content-Type": "application/json"})
        response.raise_for_status()
        return response.json()["candidates"][0]["content"]["parts"][0]["text"]
    except Exception as e:
        console.print(f"[error]❌ Gemini API Error: {e}[/error]")
        return f"# Error Generating Report\nCheck API Key."
