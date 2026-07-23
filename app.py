import os, time, json, asyncio, traceback
from fastapi import FastAPI, Request, HTTPException, Depends
from fastapi.responses import HTMLResponse, StreamingResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from utils.security import check_rate_limit, sanitize_filename
from utils.report_saver import save_report_assets
from agents.search_agent import optimize_query, fetch_articles
from agents.filter_agent import filter_and_rank_articles
from agents.scraper_agent import scrape_top_articles
from agents.report_agent import generate_report, generate_podcast_script, generate_diagram, rag_query, challenge_query

app = FastAPI(title="ThesisPilot SaaS")
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
reports_dir = os.path.join(BASE_DIR, "reports")
os.makedirs(reports_dir, exist_ok=True)

app.mount("/reports", StaticFiles(directory=reports_dir), name="reports")
templates = Jinja2Templates(directory=os.path.join(BASE_DIR, "templates"))

class InteractiveRequest(BaseModel):
    topic: str
    query: str

@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.get("/stream-research")
async def stream_research(request: Request, topic: str, language: str = "english"):
    check_rate_limit(request) # 🔴 RATE LIMIT PROTECTION
    
    async def event_generator():
        start_time = time.time()
        try:
            yield "data: [PROGRESS:10]▶ OPTIMIZING QUERY...\n\n"
            opt_topic = await asyncio.to_thread(optimize_query, topic)
            
            yield "data: [PROGRESS:30]▶ SEARCHING WEB...\n\n"
            raw = await asyncio.to_thread(fetch_articles, opt_topic)
            if not raw:
                yield "data: [PROGRESS:100]❌ No articles found.\n\n"
                return
                
            yield "data: [PROGRESS:50]▶ CREDIBILITY RANKING...\n\n"
            ranked, _, _, llm_ok = await asyncio.to_thread(filter_and_rank_articles, raw)
            
            yield "data: [PROGRESS:70]▶ DEEP SCRAPING...\n\n"
            scraped, count, real_cred = await asyncio.to_thread(scrape_top_articles, ranked)
            
            yield "data: [PROGRESS:90]▶ SYNTHESIZING ACADEMIC REPORT...\n\n"
            stats = {"scraped_success": count, "avg_credibility": real_cred, "llm_ranking_success": llm_ok}
            report, _ = await asyncio.to_thread(generate_report, opt_topic, scraped, language, stats)
            
            # 🔴 XSS & DRY SAVING PROTECTION
            assets = await asyncio.to_thread(save_report_assets, opt_topic, report, scraped)
            
            metrics = {"time": round(time.time() - start_time, 1), "scraped": count, "safe_topic": assets['safe_topic'], "md_path": assets['md_url'], "html_path": assets['html_url']}
            yield f"data: {json.dumps({'status': 'done', 'report': assets['safe_html'], 'metrics': metrics, 'topic': opt_topic})}\n\n"
            
        except Exception as e:
            yield f"data: [PROGRESS:100]❌ Error: {str(e)}\n\n"

    return StreamingResponse(event_generator(), media_type="text/event-stream")

# 🔴 PATH TRAVERSAL FIX IMPLEMENTED FOR ALL INTERACTIVE ENDPOINTS
def get_safe_context(topic: str):
    safe_topic = sanitize_filename(topic)
    path = os.path.join(reports_dir, f"{safe_topic}_context.txt")
    if not os.path.exists(path): raise HTTPException(status_code=404, detail="Context lost.")
    with open(path, "r", encoding="utf-8") as f: return f.read()

@app.post("/generate-podcast")
async def api_podcast(req: InteractiveRequest):
    return {"script": await asyncio.to_thread(generate_podcast_script, req.query)} # using query field as text for simplicity

@app.post("/generate-diagram")
async def api_diagram(req: InteractiveRequest):
    return {"mermaid": await asyncio.to_thread(generate_diagram, req.query)}

@app.post("/ask-rag")
async def api_ask_rag(req: InteractiveRequest):
    try:
        return {"answer": await asyncio.to_thread(rag_query, get_safe_context(req.topic), req.query)}
    except Exception as e: return {"error": str(e)}

@app.post("/challenge-report")
async def api_challenge(req: InteractiveRequest):
    try:
        return {"answer": await asyncio.to_thread(challenge_query, get_safe_context(req.topic), req.query)}
    except Exception as e: return {"error": str(e)}

# 🔴 MULTI-TENANT PRIVACY LEAK FIX:
@app.get("/api/report/{safe_topic}")
async def get_specific_report(safe_topic: str):
    safe_name = sanitize_filename(safe_topic)
    md_path = os.path.join(reports_dir, f"{safe_name}_report.md")
    if not os.path.exists(md_path): return {"error": "Not found"}
    with open(md_path, "r", encoding="utf-8") as f: content = f.read()
    
    import markdown, bleach # XSS Fix on read
    raw_html = markdown.markdown(content, extensions=['tables'])
    safe_html = bleach.clean(raw_html, tags=list(bleach.sanitizer.ALLOWED_TAGS)+['h1','h2','h3','table','tr','td','th'])
    return {"topic": safe_name.replace("_", " ").title(), "safe_topic": safe_name, "report": safe_html, "md_path": f"/reports/{safe_name}_report.md"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=int(os.environ.get("PORT", 8000)))
