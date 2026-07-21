import os
import time
import json
import asyncio
import traceback
import markdown
from fastapi import FastAPI, Form, Request
from fastapi.responses import HTMLResponse, StreamingResponse
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

@app.get("/stream-research")
async def stream_research(topic: str, language: str = "english"):
    async def event_generator():
        start_time = time.time()
        llm_calls = 0
        try:
            # Phase 0
            yield f"data: [PROGRESS:10]▶ PHASE 0: QUERY OPTIMIZATION...\n\n"
            optimized_topic = await asyncio.to_thread(optimize_query, topic)
            llm_calls += 1
            yield f"data: [PROGRESS:20]✨ Query optimized to: '{optimized_topic}'\n\n"
            
            # Phase 1
            yield f"data: [PROGRESS:30]▶ PHASE 1: SEARCHING WEB...\n\n"
            yield f"data: [PROGRESS:40]🔍 Tavily API fetching 80 articles...\n\n"
            raw_articles = await asyncio.to_thread(fetch_articles, optimized_topic, max_results=80)
            if not raw_articles:
                yield f"data: [PROGRESS:100]❌ No articles found for this topic.\n\n"
                return
            yield f"data: [PROGRESS:50]✅ Fetched {len(raw_articles)} raw articles successfully!\n\n"
            
            # Phase 2
            yield f"data: [PROGRESS:60]▶ PHASE 2: CREDIBILITY RANKING (NVIDIA LLM)...\n\n"
            ranked_articles, duplicates_removed, filter_calls, llm_success = await asyncio.to_thread(
                filter_and_rank_articles, raw_articles, top_n=40
            )
            llm_calls += filter_calls
            yield f"data: [PROGRESS:70]✅ Ranking done! Top {len(ranked_articles)} high-credibility sources selected.\n\n"
            
            # Phase 3
            yield f"data: [PROGRESS:80]▶ PHASE 3: ASYNC SCRAPING (Target: 15+)...\n\n"
            scraped_data, scraped_count = await asyncio.to_thread(
                scrape_top_articles, ranked_articles, min_required=15
            )
            yield f"data: [PROGRESS:90]✅ Async Scraping completed! Successful sources: {scraped_count}\n\n"
            
            # Phase 4
            yield f"data: [PROGRESS:95]▶ PHASE 4: EXTENSIVE REPORT SYNTHESIS ({language.capitalize()})...\n\n"
            
            stats_dict = {
                "scraped_success": scraped_count, 
                "avg_credibility": 8.5, 
                "duplicates_removed": duplicates_removed,
                "llm_ranking_success": llm_success
            }
            
            # 🌟 Unpacking report text and model name
            final_report, model_used = await asyncio.to_thread(
                generate_report, optimized_topic, scraped_data, language.capitalize(), stats_dict
            )
            llm_calls += 1
            yield f"data: [PROGRESS:100]✅ Comprehensive report generated via {model_used}!\n\n"
            
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
            
            final_packet = {
                "status": "done",
                "report": report_html,
                "metrics": metrics,
                "topic": optimized_topic
            }
            yield f"data: {json.dumps(final_packet)}\n\n"
            
        except Exception as e:
            err = traceback.format_exc()
            yield f"data: [PROGRESS:100]❌ Pipeline Error: {str(e)}\n\n"

    return StreamingResponse(event_generator(), media_type="text/event-stream")

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run("app:app", host="0.0.0.0", port=port)
