import os
import markdown
import bleach
from utils.security import sanitize_filename

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REPORTS_DIR = os.path.join(BASE_DIR, "reports")
os.makedirs(REPORTS_DIR, exist_ok=True)

def save_report_assets(topic: str, report_text: str, context_text: str) -> dict:
    safe_topic = sanitize_filename(topic.replace(' ', '_'))
    
    md_filename = f"{safe_topic}_report.md"
    html_filename = f"{safe_topic}_report.html"
    context_filename = f"{safe_topic}_context.txt"
    
    # Save Raw MD and Context
    with open(os.path.join(REPORTS_DIR, md_filename), "w", encoding="utf-8") as f:
        f.write(report_text)
    with open(os.path.join(REPORTS_DIR, context_filename), "w", encoding="utf-8") as f:
        f.write(context_text)

    # 🔴 CRITICAL FIX: Stored XSS Mitigation using Bleach
    raw_html = markdown.markdown(report_text, extensions=['tables', 'fenced_code'])
    allowed_tags = list(bleach.sanitizer.ALLOWED_TAGS) + ['h1','h2','h3','h4','h5','p','div','span','br','table','thead','tbody','tr','th','td','pre','code']
    safe_html_body = bleach.clean(raw_html, tags=allowed_tags)
    
    full_html = f"<html><head><meta charset='utf-8'><title>{topic}</title><style>body{{font-family: sans-serif; max-width: 900px; margin: 40px auto; line-height: 1.6;}} table{{border-collapse: collapse; width: 100%;}} th, td{{border: 1px solid #ddd; padding: 8px;}}</style></head><body>{safe_html_body}</body></html>"
    
    with open(os.path.join(REPORTS_DIR, html_filename), "w", encoding="utf-8") as f:
        f.write(full_html)
        
    return {
        "safe_topic": safe_topic,
        "md_path": f"/reports/{md_filename}",
        "html_path": f"/reports/{html_filename}",
        "safe_html_body": safe_html_body
    }
