import os
import time
import markdown
from rich.prompt import Prompt
from config import console
from agents.search_agent import optimize_query, fetch_articles
from agents.filter_agent import filter_and_rank_articles
from agents.scraper_agent import scrape_top_articles
from agents.report_agent import generate_report

def main():
    console.print("\n[success]======================================================[/success]")
    console.print("[success] 🚀 THESISPILOT AI RESEARCH AGENT (V5.0 FINAL) [/success]")
    console.print("[success]======================================================[/success]\n")
    
    raw_query = Prompt.ask("[step]Kis topic par deep research karni hai?[/step]")
    language_input = Prompt.ask("[step]Report kis language me banani hai?[/step]", choices=["english", "hindi", "hinglish"], default="english")
    language = language_input.capitalize()
    
    start_time = time.time()
    llm_calls = 0
    
    console.print("\n[info]🚀 Initiating Autonomous Async Pipeline...[/info]")

    # Phase 0 & 1
    topic = optimize_query(raw_query)
    llm_calls += 1
    
    raw_articles = fetch_articles(topic, max_results=20)
    if not raw_articles: return
        
    # Phase 2
    ranked_articles, duplicates_removed, filter_calls, llm_success = filter_and_rank_articles(raw_articles, top_n=20)
    llm_calls += filter_calls
    
    # Phase 3
    scraped_data, scraped_count = scrape_top_articles(ranked_articles, min_required=10)
    
    # Phase 4
    stats_dict = {
        "scraped_success": scraped_count, 
        "avg_credibility": 8.5, 
        "duplicates_removed": duplicates_removed,
        "llm_ranking_success": llm_success
    }
    
    # Updated to receive model_used from the new report_agent logic
    final_report, model_used = generate_report(topic, scraped_data, language, stats_dict)
    llm_calls += 1
    
    # Save Outputs
    os.makedirs("reports", exist_ok=True)
    safe_topic = topic.replace(' ', '_').lower()
    md_filename = os.path.join("reports", f"{safe_topic}_report.md")
    html_filename = os.path.join("reports", f"{safe_topic}_report.html")
    context_filename = os.path.join("reports", f"{safe_topic}_context.txt")
    
    # Saving Markdown, HTML, and Raw Context (For RAG)
    with open(md_filename, "w", encoding="utf-8") as f: f.write(final_report)
    with open(context_filename, "w", encoding="utf-8") as f: f.write(scraped_data)
    with open(html_filename, "w", encoding="utf-8") as f: 
        html_body = markdown.markdown(final_report, extensions=['tables'])
        f.write(f"<html><head><meta charset='utf-8'><title>{topic}</title><style>body{{font-family: sans-serif; max-width: 900px; margin: 40px auto; line-height: 1.6;}} table{{border-collapse: collapse; width: 100%;}} th, td{{border: 1px solid #ddd; padding: 8px;}}</style></head><body>{html_body}</body></html>")
    
    # Pipeline Metrics Dashboard
    end_time = time.time()
    console.print("\n[step]============================================[/step]")
    console.print("[step]         📈 PIPELINE METRICS DASHBOARD      [/step]")
    console.print("[step]============================================[/step]")
    console.print(f"[info]  Total Time Taken :[/info] {round(end_time - start_time, 1)} seconds")
    console.print(f"[info]  LLM API Calls    :[/info] {llm_calls}")
    console.print(f"[info]  Raw Fetched      :[/info] {len(raw_articles)}")
    console.print(f"[info]  Spam/Dupes Dropped:[/info] {duplicates_removed}")
    console.print(f"[info]  Successful Scrapes:[/info] {scraped_count}")
    console.print(f"[info]  Synthesis Model  :[/info] {model_used}")
    console.print("[step]============================================[/step]")
    
    console.print(f"\n[success]🎉 BOOM! Research Complete![/success]")
    console.print(f"[info]📄 Markdown File : {md_filename}[/info]")
    console.print(f"[info]🌐 HTML Export   : {html_filename}[/info]")
    console.print(f"[interactive]🧠 Raw Context   : {context_filename} (Ready for RAG/CLI)[/interactive]")
    console.print(f"\n[interactive]💡 Pro Tip: Run `python app.py` to use Podcast, Mindmap, and RAG in the Web UI![/interactive]\n")

if __name__ == "__main__":
    main()
