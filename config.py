import os
from rich.console import Console
from rich.theme import Theme

custom_theme = Theme({
    "info": "cyan",
    "success": "bold green",
    "warning": "bold yellow",
    "error": "bold red",
    "step": "bold magenta",
    "highlight": "bold yellow"
})

console = Console(theme=custom_theme)

# API Keys (Yahan apni actual API keys daalna)
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY", "your_tavily_key_here")
NVIDIA_API_KEY = os.getenv("NVIDIA_API_KEY", "your_nvidia_key_here")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "your_gemini_key_here")
