import os
import re
from fastapi import Request, HTTPException
import time

# Basic In-Memory Rate Limiter (For MVP SaaS)
request_history = {}
RATE_LIMIT_SECONDS = 30 # Ek IP 30 seconds me ek hi research trigger kar sakti hai

def check_rate_limit(request: Request):
    client_ip = request.client.host
    current_time = time.time()
    
    if client_ip in request_history:
        time_passed = current_time - request_history[client_ip]
        if time_passed < RATE_LIMIT_SECONDS:
            raise HTTPException(status_code=429, detail=f"Too many requests. Please wait {int(RATE_LIMIT_SECONDS - time_passed)} seconds.")
            
    request_history[client_ip] = current_time

def sanitize_filename(filename: str) -> str:
    # 🔴 CRITICAL FIX: Path Traversal
    # Only allows alphanumeric, underscores, and dashes. Strips "../" completely.
    safe_name = os.path.basename(filename)
    safe_name = re.sub(r'[^a-zA-Z0-9_\-]', '', safe_name)
    return safe_name.lower()
