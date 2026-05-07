import urllib.request
import urllib.error
import re
from src.tools.base import tool
from src.tools.registry import register_tool

@tool(name="web_search", description="Search the web using DuckDuckGo.")
def web_search(query: str, max_results: int = 5) -> str:
    try:
        from duckduckgo_search import DDGS
        results = DDGS().text(query, max_results=max_results)
        if not results:
            return "No results found."
        formatted = []
        for i, r in enumerate(results):
            formatted.append(f"{i+1}. {r.get('title', '')}\nURL: {r.get('href', '')}\n{r.get('body', '')}\n")
        return "\n".join(formatted)
    except ImportError:
        return "Error: duckduckgo-search package is not installed."
    except Exception as e:
        return f"Error performing web search: {e}"

@tool(name="web_fetch", description="Fetch text content from a URL.")
def web_fetch(url: str) -> str:
    try:
        req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
        with urllib.request.urlopen(req, timeout=30) as response:
            html = response.read().decode('utf-8')
            # Extremely simplistic HTML to text stripping
            body = re.search(r'<body[^>]*>(.*?)</body>', html, re.IGNORECASE | re.DOTALL)
            text = body.group(1) if body else html
            text = re.sub(r'<script[^>]*>.*?</script>', '', text, flags=re.IGNORECASE | re.DOTALL)
            text = re.sub(r'<style[^>]*>.*?</style>', '', text, flags=re.IGNORECASE | re.DOTALL)
            text = re.sub(r'<[^>]+>', ' ', text)
            text = re.sub(r'\s+', ' ', text).strip()
            return text[:10000] # Limit to 10k chars to avoid blowing up context
    except Exception as e:
        return f"Error fetching URL: {e}"

for t in [web_search, web_fetch]:
    register_tool(t)
