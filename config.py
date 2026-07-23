import os
import logging
from rich.console import Console
from rich.theme import Theme

# Centralized Models
MODEL_HEAVY = "meta/llama-3.1-70b-instruct"
MODEL_LIGHT = "meta/llama-3.1-8b-instruct"

# API Keys
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY", "your_tavily_key_here")
NVIDIA_API_KEY = os.getenv("NVIDIA_API_KEY", "your_nvidia_key_here")

# Rich Console
custom_theme = Theme({
    "info": "cyan", "success": "bold green", "warning": "bold yellow", 
    "error": "bold red", "step": "bold magenta", "highlight": "bold yellow", "interactive": "bold cyan"
})
console = Console(theme=custom_theme)

# Standard Logging for Production
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("ThesisPilot")
