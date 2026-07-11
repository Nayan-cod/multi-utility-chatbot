# C:\langgraph\scholar_server.py
from mcp.server.fastmcp import FastMCP
from serpapi import GoogleSearch
import os
from dotenv import load_dotenv

# Load the key from your .env file
load_dotenv()
SERPAPI_KEY = os.getenv("SERPAPI_API_KEY")

mcp = FastMCP("scholar_tools")


@mcp.tool()
def search_google_scholar(query: str, limit: int = 1) -> str:
    """
    Search Google Scholar using SerpApi (Reliable & No Blocks).
    Returns title, link, and a very short summary/snippet to conserve tokens.
    """
    if not SERPAPI_KEY:
        return "Error: SERPAPI_API_KEY not found in .env file."

    try:
        # Cap the limit to maximum 3 to prevent token blowup
        search_limit = min(max(1, limit), 3)
        params = {
            "engine": "google_scholar",
            "q": query,
            "api_key": SERPAPI_KEY,
            "num": search_limit,
        }
        search = GoogleSearch(params)
        results = search.get_dict().get("organic_results", [])

        output = []
        for item in results:
            title = item.get("title")
            link = item.get("link")
            snippet = item.get("snippet", "")
            # Truncate snippet to 120 chars to save tokens
            short_snippet = (snippet[:120] + "...") if len(snippet) > 120 else snippet
            pub_info = item.get("publication_info", {}).get("summary", "")

            # Compact output structure: Link and some basic info
            parts = [f"Title: {title}"]
            if link:
                parts.append(f"Link: {link}")
            if pub_info:
                parts.append(f"Info: {pub_info}")
            if short_snippet:
                parts.append(f"Snippet: {short_snippet}")
            
            output.append("\n".join(parts))

        return "\n---\n".join(output) if output else "No results found."

    except Exception as e:
        return f"SerpApi Error: {str(e)}"


if __name__ == "__main__":
    mcp.run()
