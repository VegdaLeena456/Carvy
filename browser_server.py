# browser_server.py
from mcp.server import MCPServer
import webbrowser
from urllib.parse import quote_plus

# Create the MCP server
mcp = MCPServer("Browser Tools")

@mcp.tool()
def open_search(query: str) -> str:
    """
    Use this tool ONLY when the user wants to search something on Google.
    Do NOT use this for normal questions.
    """
    search_url = f"https://www.google.com/search?q={quote_plus(query)}"
    webbrowser.open(search_url)
    return f"Opened browser and searched for: {query}"


@mcp.tool()
def open_website(url: str) -> str:
    """
    Use this tool ONLY when the user wants to open a website.
    Do NOT use this for normal questions.
    """
    if not url.startswith(("http://", "https://")):
        url = "https://" + url
    webbrowser.open(url)
    return f"Opened website: {url}"


if __name__ == "__main__":
    mcp.run()   # default is stdio