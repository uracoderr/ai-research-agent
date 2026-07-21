import requests
import json
from config import console, GEMINI_API_KEY, NVIDIA_API_KEY

def generate_report(topic: str, scraped_text: str, language: str, stats: dict) -> tuple:
    console.print(f"\n[step]▶ PHASE 4: EXTENSIVE REPORT GENERATION ({language.upper()})[/step]")
    
    # Realistic Dynamic Confidence Calculation
    base_score = (stats['avg_credibility'] / 10) * 60
    volume_score = min(30, (stats['scraped_success'] / 15) * 30)
    llm_penalty = 0 if stats['llm_ranking_success'] else 15
    confidence_score = max(50, min(98, int(base_score + volume_score - llm_penalty)))

    # 🌟 Ultra-strict language instruction builder with examples
    lang_lower = language.lower()
    if "hinglish" in lang_lower:
        lang_instruction = (
            "🚨 MANDATORY LANGUAGE LOCK: You MUST write the ENTIRE report strictly in conversational HINGLISH "
            "using ONLY Latin/Roman alphabet (English letters). DO NOT use Devanagari script. DO NOT write standard formal English. "
            "Mix Hindi conversational style naturally with technical English terms. "
            "Example style to follow: 'Yeh report AI ke current trends aur future predictions par based hai. "
            "Market me bahut saare advancements dekhne ko mil rahe hain...' Write EVERY SINGLE SECTION in this style."
        )
    elif "hindi" in lang_lower:
        lang_instruction = (
            "🚨 MANDATORY LANGUAGE LOCK: You MUST write the ENTIRE report in pure HINDI "
            "using Devanagari script (हिंदी लिपि). Sabhi sections hindi me hone chahiye."
        )
    else:
        lang_instruction = "Write the report in deep, professional, comprehensive English."

    prompt = f"""
    You are an elite Chief Industry Analyst writing an extensive, highly comprehensive, long-form research report for 2026. 
    Do NOT write a short summary. Expand thoroughly on every section with deep granularity, specific market context, data points, and exhaustive details.
    
    Topic: '{topic}'
    
    {lang_instruction}
    
    Use the provided scraped data thoroughly. Cite sources appropriately using inline tags like [Source Name].
    
    REQUIRED DEEP-DIVE REPORT STRUCTURE (Make each section long, detailed, and rich in analysis):
    
    # [Topic] - Comprehensive Industry Intelligence Report 2026
    
    ## 📊 Executive Dashboard
    - **Confidence Score:** {confidence_score}%
    - **Sources Processed & Scraped:** {stats['scraped_success']}
    - **Avg Source Credibility:** {stats['avg_credibility']}/10
    - **Junk/Duplicates Filtered:** {stats['duplicates_removed']}
    - **Ranking Engine Status:** NVIDIA Llama-3.1
    - **Synthesis Model:** [MODEL_TAG]
    
    ## 📑 Executive Summary & Deep Market Context
    (Provide a lengthy, highly detailed breakdown of the core landscape, macro shifts, and immediate takeaways. Expand into multiple paragraphs.)
    
    ## 🔬 Comprehensive Technical & Market Analysis
    (Provide an exhaustive deep-dive analysis. Break down core pillars, technical innovations, market drivers, contrasting viewpoints, and contradictions found across sources. Use multiple subsections or detailed paragraphs.)
    
    ## 📈 Granular Opportunity & Risk Matrix
    (Provide an extensive breakdown of market opportunities vs significant systemic risks. Include structured markdown tables or bulleted analysis.)
    
    ## 🚀 Long-term Strategic Predictions & Actionable Roadmap
    (Provide detailed 2 to 5 year timelines, predictions, and strategic advice for stakeholders.)
    
    ## 📚 Weighted Sources & Credibility Breakdown
    (List and evaluate the key sources utilized in this synthesis.)

    Scraped Data Context:
    {scraped_text}
    """

    model_used = "Unknown"
    report_text = ""

    # Try Gemini First
    gemini_url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash:generateContent?key={GEMINI_API_KEY}"
    payload = {"contents": [{"parts": [{"text": prompt}]}]}

    try:
        response = requests.post(gemini_url, json=payload, headers={"Content-Type": "application/json"}, timeout=45)
        if response.status_code == 200:
            report_text = response.json()["candidates"][0]["content"]["parts"][0]["text"]
            model_used = "Gemini 2.5 Flash"
        else:
            console.print(f"[warning]⚠ Gemini Quota/Error ({response.status_code}). Switching to NVIDIA Llama...[/warning]")
    except Exception as e:
        console.print(f"[warning]⚠ Gemini Failed: {e}. Switching to NVIDIA Llama...[/warning]")

    # Fallback / Primary via NVIDIA Llama if Gemini failed
    if not report_text:
        nvidia_url = "https://integrate.api.nvidia.com/v1/chat/completions"
        headers = {"Authorization": f"Bearer {NVIDIA_API_KEY}", "Content-Type": "application/json"}
        nvidia_payload = {
            "model": "meta/llama-3.1-8b-instruct",
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.3,
            "max_tokens": 4096
        }

        try:
            res = requests.post(nvidia_url, headers=headers, json=nvidia_payload, timeout=60)
            res.raise_for_status()
            report_text = res.json()["choices"][0]["message"]["content"]
            model_used = "NVIDIA Llama-3.1 8B"
        except Exception as e:
            report_text = f"# Error Generating Report\nAll LLM synthesis failed. Details: {str(e)}"
            model_used = "Failed"

    # Inject model used into the report's executive dashboard placeholder
    report_text = report_text.replace("[MODEL_TAG]", model_used)
    return report_text, model_used
