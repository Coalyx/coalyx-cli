"""Web tools with SSRF protection, proper HTML parsing, and content tagging.

Security measures:
- URL scheme allow-list (only http/https)
- DNS resolution check against private/internal IP ranges
- Content tagged with [WEB_CONTENT] markers for prompt injection awareness
"""

import ipaddress
import socket
import urllib.parse
import urllib.request
import urllib.error
from html.parser import HTMLParser
from typing import List

from src.tools.base import tool
from src.tools.registry import register_tool

# ---------------------------------------------------------------------------
# SSRF protection
# ---------------------------------------------------------------------------

ALLOWED_SCHEMES = {"http", "https"}

BLOCKED_NETWORKS = [
    ipaddress.ip_network("127.0.0.0/8"),       # loopback
    ipaddress.ip_network("10.0.0.0/8"),         # private class A
    ipaddress.ip_network("172.16.0.0/12"),      # private class B
    ipaddress.ip_network("192.168.0.0/16"),     # private class C
    ipaddress.ip_network("169.254.0.0/16"),     # link-local / cloud metadata
    ipaddress.ip_network("0.0.0.0/8"),          # "this" network
    ipaddress.ip_network("::1/128"),            # IPv6 loopback
    ipaddress.ip_network("fc00::/7"),           # IPv6 unique-local
    ipaddress.ip_network("fe80::/10"),          # IPv6 link-local
]


class URLSecurityError(Exception):
    """Raised when a URL fails SSRF validation."""


def validate_url(url: str) -> str:
    """Validate *url* against SSRF attacks.

    Checks:
    1. Scheme must be http or https.
    2. Hostname must resolve to a public (non-private) IP address.

    Returns the validated URL on success.

    Raises:
        URLSecurityError: if the URL is blocked.
    """
    parsed = urllib.parse.urlparse(url)

    if parsed.scheme not in ALLOWED_SCHEMES:
        raise URLSecurityError(
            f"Blocked URL scheme '{parsed.scheme}'. Only {ALLOWED_SCHEMES} allowed."
        )

    hostname = parsed.hostname
    if not hostname:
        raise URLSecurityError("URL has no hostname.")

    try:
        addr_infos = socket.getaddrinfo(hostname, None)
    except socket.gaierror:
        raise URLSecurityError(f"Cannot resolve hostname '{hostname}'.")

    for info in addr_infos:
        ip_str = info[4][0]
        try:
            addr = ipaddress.ip_address(ip_str)
        except ValueError:
            continue
        for net in BLOCKED_NETWORKS:
            if addr in net:
                raise URLSecurityError(
                    f"Blocked: '{hostname}' resolves to private/internal address {addr}."
                )

    return url


# ---------------------------------------------------------------------------
# HTML-to-text parser (stdlib, no regex)
# ---------------------------------------------------------------------------

class _HTMLTextExtractor(HTMLParser):
    """Minimal HTML→text converter using stdlib HTMLParser."""

    SKIP_TAGS = {"script", "style", "noscript", "svg", "head"}

    def __init__(self):
        super().__init__()
        self._pieces: List[str] = []
        self._skip_depth: int = 0

    def handle_starttag(self, tag: str, attrs):
        if tag.lower() in self.SKIP_TAGS:
            self._skip_depth += 1

    def handle_endtag(self, tag: str):
        if tag.lower() in self.SKIP_TAGS and self._skip_depth > 0:
            self._skip_depth -= 1

    def handle_data(self, data: str):
        if self._skip_depth == 0:
            stripped = data.strip()
            if stripped:
                self._pieces.append(stripped)

    def get_text(self) -> str:
        return " ".join(self._pieces)


def html_to_text(html: str) -> str:
    """Convert HTML to plain text using stdlib HTMLParser."""
    parser = _HTMLTextExtractor()
    try:
        parser.feed(html)
    except Exception:
        # Graceful fallback — return raw html stripped of tags
        import re
        return re.sub(r"<[^>]+>", " ", html)
    return parser.get_text()


# ---------------------------------------------------------------------------
# Tools
# ---------------------------------------------------------------------------

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
        return "[WEB_CONTENT]\n" + "\n".join(formatted) + "\n[/WEB_CONTENT]"
    except ImportError:
        return "Error: duckduckgo-search package is not installed."
    except Exception as e:
        return f"Error performing web search: {e}"

@tool(name="web_fetch", description="Fetch text content from a URL.")
def web_fetch(url: str) -> str:
    try:
        validated_url = validate_url(url)
    except URLSecurityError as e:
        return f"Security error: {e}"

    try:
        req = urllib.request.Request(validated_url, headers={'User-Agent': 'Mozilla/5.0'})
        with urllib.request.urlopen(req, timeout=30) as response:
            # Respect encoding from Content-Type header
            content_type = response.headers.get("Content-Type", "")
            charset = "utf-8"
            if "charset=" in content_type:
                charset = content_type.split("charset=")[-1].split(";")[0].strip()

            try:
                html = response.read().decode(charset)
            except (UnicodeDecodeError, LookupError):
                html = response.read().decode("utf-8", errors="replace")

            text = html_to_text(html)
            truncated = text[:10000]  # Limit to 10k chars

            return (
                f"[WEB_CONTENT source=\"{validated_url}\"]\n"
                f"{truncated}\n"
                f"[/WEB_CONTENT]"
            )
    except Exception as e:
        return f"Error fetching URL: {e}"

for t in [web_search, web_fetch]:
    register_tool(t)
