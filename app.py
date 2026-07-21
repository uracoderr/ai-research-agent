import os
import time
import json
import traceback
import markdown
from fastapi import FastAPI, Form, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles

from agents.search_agent import optimize_query, fetch_articles
from agents.filter_agent import filter_and_rank_articles
from agents.scraper_agent import scrape_top_articles
from agents.report_agent import generate_report

app = FastAPI(title="AI Research Agent Web")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
templates_dir = os.path.join(BASE_DIR, "templates")
templates = Jinja2Templates(directory=templates_dir)

reports_dir = os.path.join(BASE_DIR, "reports")
os.makedirs(reports_dir, exist_ok=True)
app.mount("/reports", StaticFiles(directory=reports_dir), name="reports")

def render_template(request: Request, template_name: str, context: dict):
    try:
        return templates.TemplateResponse(request, template_name, context)
    except TypeError:
        context["request"] = request
        return templates.TemplateResponse(template_name, context)

@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return render_template(request, "index.html", {"result": None, "error": None})

@app.post("/run-research")
async def run_research(topic: str = Form(...), language: str = Form(...)):
    start_time = time.time()
    llm_calls = 0
    try:
        # Phase 0: Query Optimization
        optimized_topic = optimize_query(topic)
        llm_calls += 1
        
        # Phase 1: Search
        raw_articles = fetch_articles(optimized_topic, max_results=80)
        if not raw_articles:
            return JSONResponse({"status": "error", "message": "No articles found for this topic."})
        
        # Phase 2: Filtering & Ranking (70B)
        ranked_articles, duplicates_removed, filter_calls, llm_success = filter_and_rank_articles(raw_articles, top_n=40)
        llm_calls += filter_calls
        
        # Phase 3: Scraping
        scraped_data, scraped_count = scrape_top_articles(ranked_articles, min_required=15)
        
        # Phase 4: Report Synthesis
        stats_dict = {
            "scraped_success": scraped_count, 
            "avg_credibility": 8.5, 
            "duplicates_removed": duplicates_removed,
            "llm_ranking_success": llm_success
        }
        
        final_report, model_used = generate_report(optimized_topic, scraped_data, language.capitalize(), stats_dict)
        llm_calls += 1
        
        # Save files
        safe_topic = optimized_topic.replace(' ', '_').lower()
        md_filename = os.path.join("reports", f"{safe_topic}_report.md")
        html_filename = os.path.join("reports", f"{safe_topic}_report.html")
        
        with open(os.path.join(BASE_DIR, md_filename), "w", encoding="utf-8") as f:
            f.write(final_report)
            
        html_content = f"<html><head><meta charset='utf-8'><title>{optimized_topic}</title></head><body>{markdown.markdown(final_report, extensions=['tables'])}</body></html>"
        with open(os.path.join(BASE_DIR, html_filename), "w", encoding="utf-8") as f:
            f.write(html_content)
            
        report_html = markdown.markdown(final_report, extensions=['tables', 'fenced_code'])
        
        end_time = time.time()
        metrics = {
            "time": round(end_time - start_time, 1),
            "calls": llm_calls,
            "fetched": len(raw_articles),
            "scraped": scraped_count,
            "md_path": f"/{md_filename}",
            "html_path": f"/{html_filename}"
        }
        
        return {
            "status": "success",
            "topic": optimized_topic,
            "report": report_html,
            "metrics": metrics
        }
        
    except Exception as e:
        return JSONResponse({"status": "error", "message": str(e)})

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run("app:app", host="0.0.0.0", port=port)
