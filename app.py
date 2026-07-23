import os
import time
import json
import asyncio
import traceback
import markdown
from fastapi import FastAPI, Form, Request, HTTPException
from fastapi.responses import HTMLResponse, StreamingResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from agents.search_agent import optimize_query, fetch_articles
from agents.filter_agent import filter_and_rank_articles
from agents.scraper_agent import scrape_top_articles
from agents.report_agent import generate_report, generate_podcast_script, generate_diagram, rag_query, challenge_query

app = FastAPI(title="ThesisPilot AI Web")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
templates_dir = os.path.join(BASE_DIR, "templates")
templates = Jinja2Templates(directory=templates_dir)

reports_dir = os.path.join(BASE_DIR, "reports")
os.makedirs(reports_dir, exist_ok=True)
app.mount("/reports", StaticFiles(directory=reports_dir), name="reports")

# Request Models
class PodcastRequest(BaseModel):
    report_text: str

class InteractiveRequest(BaseModel):
    topic: str
    query: str

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
            
            # Phase 1 - Changed to 20 articles limit
            yield f"data: [PROGRESS:30]▶ PHASE 1: SEARCHING WEB...\n\n"
            yield f"data: [PROGRESS:40]🔍 Tavily API fetching 20 articles...\n\n"
            await asyncio.sleep(0.5)
            raw_articles = await asyncio.to_thread(fetch_articles, optimized_topic, max_results=20)
            if not raw_articles:
                yield f"data: [PROGRESS:100]❌ No articles found for this topic.\n\n"
                return
            yield f"data: [PROGRESS:50]✅ Fetched {len(raw_articles)} raw articles successfully!\n\n"
            
            # Phase 2 - Ranking limit adjusted to top 20
            yield f"data: [PROGRESS:60]▶ PHASE 2: CREDIBILITY RANKING...\n\n"
            ranked_articles, duplicates_removed, filter_calls, llm_success = await asyncio.to_thread(
                filter_and_rank_articles, raw_articles, top_n=20
            )
            llm_calls += filter_calls
            yield f"data: [PROGRESS:70]✅ Ranking done! High-credibility sources selected.\n\n"
            
            # Phase 3 - Scrape target adjusted to 10
            yield f"data: [PROGRESS:80]▶ PHASE 3: ASYNC SCRAPING (Target: 10+)...\n\n"
            scraped_data, scraped_count = await asyncio.to_thread(
                scrape_top_articles, ranked_articles, min_required=10
            )
            yield f"data: [PROGRESS:90]✅ Async Scraping completed! Sources: {scraped_count}\n\n"
            
            # Phase 4 - Synthesis
            yield f"data: [PROGRESS:95]▶ PHASE 4: EXTENSIVE REPORT SYNTHESIS...\n\n"
            stats_dict = {
                "scraped_success": scraped_count, 
                "avg_credibility": 8.5, 
                "duplicates_removed": duplicates_removed,
                "llm_ranking_success": llm_success
            }
            
            final_report, model_used = await asyncio.to_thread(
                generate_report, optimized_topic, scraped_data, language.capitalize(), stats_dict
            )
            llm_calls += 1
            yield f"data: [PROGRESS:100]✅ Comprehensive report generated!\n\n"
            
            # Save files & context for RAG
            safe_topic = optimized_topic.replace(' ', '_').lower()
            md_filename = os.path.join("reports", f"{safe_topic}_report.md")
            html_filename = os.path.join("reports", f"{safe_topic}_report.html")
            context_filename = os.path.join("reports", f"{safe_topic}_context.txt")
            
            with open(os.path.join(BASE_DIR, md_filename), "w", encoding="utf-8") as f:
                f.write(final_report)
            with open(os.path.join(BASE_DIR, context_filename), "w", encoding="utf-8") as f:
                f.write(scraped_data) # SAVING CONTEXT FOR RAG & CHALLENGE
                
            html_content = f"<html><head><meta charset='utf-8'><title>{optimized_topic}</title><style>body{{font-family: sans-serif; max-width: 900px; margin: 40px auto; line-height: 1.6;}} table{{border-collapse: collapse; width: 100%;}} th, td{{border: 1px solid #ddd; padding: 8px;}}</style></head><body>{markdown.markdown(final_report, extensions=['tables'])}</body></html>"
            with open(os.path.join(BASE_DIR, html_filename), "w", encoding="utf-8") as f:
                f.write(html_content)
                
            report_html = markdown.markdown(final_report, extensions=['tables', 'fenced_code'])
            
            metrics = {
                "time": round(time.time() - start_time, 1),
                "calls": llm_calls,
                "fetched": len(raw_articles),
                "scraped": scraped_count,
                "md_path": f"/{md_filename}",
                "html_path": f"/{html_filename}",
                "safe_topic": safe_topic
            }
            
            yield f"data: {json.dumps({'status': 'done', 'report': report_html, 'metrics': metrics, 'topic': optimized_topic})}\n\n"
            
        except Exception as e:
            err = traceback.format_exc()
            yield f"data: [PROGRESS:100]❌ Pipeline Error: {str(e)}\n\n"

    return StreamingResponse(event_generator(), media_type="text/event-stream")

# --- NEW FEATURES APIs ---

@app.post("/generate-podcast")
async def api_podcast(req: PodcastRequest):
    try:
        script = await asyncio.to_thread(generate_podcast_script, req.report_text)
        return {"script": script}
    except Exception as e:
        return {"error": str(e)}

@app.post("/generate-diagram")
async def api_diagram(req: PodcastRequest):
    try:
        mermaid_code = await asyncio.to_thread(generate_diagram, req.report_text)
        return {"mermaid": mermaid_code}
    except Exception as e:
        return {"error": str(e)}

@app.post("/ask-rag")
async def api_ask_rag(req: InteractiveRequest):
    context_path = os.path.join(reports_dir, f"{req.topic}_context.txt")
    if not os.path.exists(context_path):
        return {"answer": "Context lost. Please regenerate the report."}
    try:
        with open(context_path, "r", encoding="utf-8") as f:
            context = f.read()
        answer = await asyncio.to_thread(rag_query, context, req.query)
        return {"answer": markdown.markdown(answer)}
    except Exception as e:
        return {"error": str(e)}

@app.post("/challenge-report")
async def api_challenge(req: InteractiveRequest):
    context_path = os.path.join(reports_dir, f"{req.topic}_context.txt")
    if not os.path.exists(context_path):
        return {"answer": "Context lost. Cannot debate without raw data."}
    try:
        with open(context_path, "r", encoding="utf-8") as f:
            context = f.read()
        answer = await asyncio.to_thread(challenge_query, context, req.query)
        return {"answer": markdown.markdown(answer)}
    except Exception as e:
        return {"error": str(e)}


# 📜 API 1: Get list of all past research reports
@app.get("/api/history")
async def get_history():
    history_list = []
    if os.path.exists(reports_dir):
        for file in os.listdir(reports_dir):
            if file.endswith("_report.md"):
                safe_topic = file.replace("_report.md", "")
                display_title = safe_topic.replace("_", " ").title()
                history_list.append({
                    "safe_topic": safe_topic,
                    "title": display_title
                })
    # Latest research ko pehle dikhane ke liye reverse kar rahe hain
    return {"history": history_list[::-1]}

# 📖 API 2: Fetch a specific past report by topic
@app.get("/api/report/{safe_topic}")
async def get_past_report(safe_topic: str):
    md_filename = os.path.join(reports_dir, f"{safe_topic}_report.md")
    if not os.path.exists(md_filename):
        return {"error": "Report not found"}
    
    with open(md_filename, "r", encoding="utf-8") as f:
        md_content = f.read()
    
    report_html = markdown.markdown(md_content, extensions=['tables', 'fenced_code'])
    
    return {
        "topic": safe_topic.replace("_", " ").title(),
        "safe_topic": safe_topic,
        "report": report_html,
        "md_path": f"/reports/{safe_topic}_report.md",
        "html_path": f"/reports/{safe_topic}_report.html"
    }


if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run("app:app", host="0.0.0.0", port=port)


