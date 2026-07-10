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
def search_google_scholar(query: str, limit: int = 5) -> str:
    """
    Search Google Scholar using SerpApi (Reliable & No Blocks).
    """
    if not SERPAPI_KEY:
        return "Error: SERPAPI_API_KEY not found in .env file."

    try:
        params = {
            "engine": "google_scholar",
            "q": query,
            "api_key": SERPAPI_KEY,
            "num": limit,
        }
        search = GoogleSearch(params)
        results = search.get_dict().get("organic_results", [])

        output = []
        for item in results:
            title = item.get("title")
            link = item.get("link")
            snippet = item.get("snippet")
            pub_info = item.get("publication_info", {}).get("summary", "")

            output.append(
                f"Title: {title}\nInfo: {pub_info}\nLink: {link}\nAbstract: {snippet}\n---"
            )

        return "\n".join(output) if output else "No results found."

    except Exception as e:
        return f"SerpApi Error: {str(e)}"


if __name__ == "__main__":
    mcp.run()
