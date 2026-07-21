import os
import time
import traceback
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

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
templates_dir = os.path.join(BASE_DIR, "templates")

# Debugging print for Render logs
print(f"--- BASE_DIR: {BASE_DIR}")
print(f"--- Templates Directory Exists: {os.path.exists(templates_dir)}")

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
    try:
        return render_template(request, "index.html", {"result": None, "error": None})
    except Exception as e:
        # 🌟 Ab agar koi error hoga toh screen par saaf-saaf dikh jayega!
        err_msg = traceback.format_exc()
        return HTMLResponse(content=f"<h3 style='color:red;'>Detailed Server Error:</h3><pre>{err_msg}</pre>", status_code=500)

@app.post("/research", response_class=HTMLResponse)
async def run_research(request: Request, topic: str = Form(...), language: str = Form(...)):
    start_time = time.time()
    llm_calls = 0
    
    try:
        optimized_topic = optimize_query(topic)
        llm_calls += 1
        
        raw_articles = fetch_articles(optimized_topic, max_results=80)
        if not raw_articles:
            return render_template(request, "index.html", {
                "result": None, 
                "error": "No articles found for this topic. Try something else."
            })
            
        ranked_articles, duplicates_removed, filter_calls, llm_success = filter_and_rank_articles(raw_articles, top_n=40)
        llm_calls += filter_calls
        
        scraped_data, scraped_count = scrape_top_articles(ranked_articles, min_required=15)
        
        stats_dict = {
            "scraped_success": scraped_count, 
            "avg_credibility": 8.5, 
            "duplicates_removed": duplicates_removed,
            "llm_ranking_success": llm_success
        }
        final_report = generate_report(optimized_topic, scraped_data, language.capitalize(), stats_dict)
        llm_calls += 1
        
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
        
        report_html = markdown.markdown(final_report, extensions=['tables', 'fenced_code'])
        
        return render_template(request, "index.html", {
            "result": report_html, 
            "metrics": metrics,
            "topic": optimized_topic,
            "error": None
        })
        
    except Exception as e:
        err_msg = traceback.format_exc()
        return HTMLResponse(content=f"<h3 style='color:red;'>Pipeline Error Traceback:</h3><pre>{err_msg}</pre>", status_code=500)

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run("app:app", host="0.0.0.0", port=port)
