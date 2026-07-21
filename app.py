import os
import time
import markdown
from fastapi import FastAPI, Form, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles

# Apne modular research agents import kar rahe hain
from agents.search_agent import optimize_query, fetch_articles
from agents.filter_agent import filter_and_rank_articles
from agents.scraper_agent import scrape_top_articles
from agents.report_agent import generate_report

app = FastAPI(title="AI Research Agent Web")

# 🌟 Safe Absolute Paths for Render / Cloud Deployment
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
templates = Jinja2Templates(directory=os.path.join(BASE_DIR, "templates"))

reports_dir = os.path.join(BASE_DIR, "reports")
os.makedirs(reports_dir, exist_ok=True)
app.mount("/reports", StaticFiles(directory=reports_dir), name="reports")

@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request, "result": None, "error": None})

@app.post("/research", response_class=HTMLResponse)
async def run_research(request: Request, topic: str = Form(...), language: str = Form(...)):
    start_time = time.time()
    llm_calls = 0
    
    try:
        # Phase 0 & 1: Query optimization & Search
        optimized_topic = optimize_query(topic)
        llm_calls += 1
        
        raw_articles = fetch_articles(optimized_topic, max_results=80)
        if not raw_articles:
            return templates.TemplateResponse("index.html", {
                "request": request, 
                "result": None, 
                "error": "No articles found for this topic. Try something else."
            })
            
        # Phase 2: Credibility Ranking via NVIDIA Llama
        ranked_articles, duplicates_removed, filter_calls, llm_success = filter_and_rank_articles(raw_articles, top_n=40)
        llm_calls += filter_calls
        
        # Phase 3: Async Scraping with auto-refill threshold
        scraped_data, scraped_count = scrape_top_articles(ranked_articles, min_required=15)
        
        # Phase 4: Gemini Report Synthesis
        stats_dict = {
            "scraped_success": scraped_count, 
            "avg_credibility": 8.5, 
            "duplicates_removed": duplicates_removed,
            "llm_ranking_success": llm_success
        }
        final_report = generate_report(optimized_topic, scraped_data, language.capitalize(), stats_dict)
        llm_calls += 1
        
        # Save Markdown and HTML reports locally for downloads
        safe_topic = optimized_topic.replace(' ', '_').lower()
        md_filename = os.path.join("reports", f"{safe_topic}_report.md")
        html_filename = os.path.join("reports", f"{safe_topic}_report.html")
        
        with open(os.path.join(BASE_DIR, md_filename), "w", encoding="utf-8") as f:
            f.write(final_report)
            
        html_content = f"""
        <html><head><meta charset='utf-8'><title>{optimized_topic} - AI Report</title>
        <style>body{{font-family: Arial, sans-serif; max-width: 900px; margin: 40px auto; line-height: 1.6; padding: 0 20px; color: #333;}} 
        h1, h2, h3{{color: #1e293b;}} table{{border-collapse: collapse; width: 100%; margin: 20px 0;}} 
        th, td{{border: 1px solid #cbd5e1; padding: 10px; text-align: left;}} th{{background-color: #f1f5f9;}}</style>
        </head><body>{markdown.markdown(final_report, extensions=['tables'])}</body></html>
        """
        with open(os.path.join(BASE_DIR, html_filename), "w", encoding="utf-8") as f:
            f.write(html_content)
            
        end_time = time.time()
        metrics = {
            "time": round(end_time - start_time, 1),
            "calls": llm_calls,
            "fetched": len(raw_articles),
            "scraped": scraped_count,
            "md_path": f"/{md_filename}",
            "html_path": f"/{html_filename}"
        }
        
        # Convert report to HTML for clean browser rendering inside UI
        report_html = markdown.markdown(final_report, extensions=['tables', 'fenced_code'])
        
        return templates.TemplateResponse("index.html", {
            "request": request, 
            "result": report_html, 
            "metrics": metrics,
            "topic": optimized_topic,
            "error": None
        })
        
    except Exception as e:
        return templates.TemplateResponse("index.html", {
            "request": request, 
            "result": None, 
            "error": f"Pipeline Error: {str(e)}"
        })

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run("app:app", host="0.0.0.0", port=port)
