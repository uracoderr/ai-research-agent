import os
from rich.console import Console
from rich.theme import Theme

custom_theme = Theme({
    "info": "cyan",
    "success": "bold green",
    "warning": "bold yellow",
    "error": "bold red",
    "step": "bold magenta",
    "highlight": "bold yellow",
    "interactive": "bold cyan"
})

console = Console(theme=custom_theme)

# API Keys (Gemini Completely Removed)
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY", "your_tavily_key_here")
NVIDIA_API_KEY = os.getenv("NVIDIA_API_KEY", "your_nvidia_key_here")
