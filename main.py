import os, time
from rich.prompt import Prompt
from config import console
from utils.report_saver import save_report_assets
from agents.search_agent import optimize_query, fetch_articles
from agents.filter_agent import filter_and_rank_articles
from agents.scraper_agent import scrape_top_articles
from agents.report_agent import generate_report

def main():
    console.print("\n[success]🚀 THESISPILOT AI - SAAS CLI[/success]\n")
    topic = optimize_query(Prompt.ask("[step]Topic?[/step]"))
    lang = Prompt.ask("[step]Language?[/step]", choices=["english", "hindi", "hinglish"], default="english").capitalize()
    
    start = time.time()
    raw = fetch_articles(topic)
    if not raw: return
    
    ranked, _, _, llm_ok = filter_and_rank_articles(raw)
    # Get Real Avg Credibility
    scraped, count, real_cred = scrape_top_articles(ranked) 
    
    stats = {"scraped_success": count, "avg_credibility": real_cred, "llm_ranking_success": llm_ok}
    report, model = generate_report(topic, scraped, lang, stats)
    
    # Use DRY Saver
    assets = save_report_assets(topic, report, scraped)
    
    console.print(f"\n[success]🎉 Time: {round(time.time()-start, 1)}s | MD: {assets['md_path']} | Credibility: {real_cred}/10[/success]")

if __name__ == "__main__":
    main()
