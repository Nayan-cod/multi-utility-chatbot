import asyncio
import threading
import requests
import os
import tempfile
import aiosqlite
from dotenv import load_dotenv
from typing import Annotated, Any, Dict, Optional, TypedDict
from contextlib import asynccontextmanager

from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
from langchain_core.messages import SystemMessage, BaseMessage
from langchain_core.tools import tool, StructuredTool
from langchain_community.tools import DuckDuckGoSearchRun
from langchain_huggingface import (
    ChatHuggingFace,
    HuggingFaceEndpoint,
    HuggingFaceEndpointEmbeddings,
)
from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS

load_dotenv()

# -------------------
# 0. Async Runner Setup
# -------------------
_ASYNC_LOOP = asyncio.new_event_loop()
_ASYNC_THREAD = threading.Thread(target=_ASYNC_LOOP.run_forever, daemon=True)
_ASYNC_THREAD.start()

_THREAD_RETRIEVERS: Dict[str, Any] = {}
_THREAD_METADATA: Dict[str, dict] = {}


def run_async(coro):
    """Helper to run async code on the background thread."""
    return asyncio.run_coroutine_threadsafe(coro, _ASYNC_LOOP).result()


# -------------------
# 1. MCP Client Setup (Updated)
# -------------------
client = MultiServerMCPClient(
    {
        # Existing Arithmetic Server
        "arithmetic": {
            "transport": "stdio",
            "command": "python",
            "args": [r".\mcp_server.py"],
        },
        # NEW: Google Scholar Server
        "scholar": {
            "transport": "stdio",
            "command": "python",
            "args": [r".\scholar_server.py"],  # <--- Ensure path is correct
        },
    }
)


async def _init_mcp_client():
    # This fetches tools from ALL connected servers (arithmetic + scholar)
    return await client.get_tools()


# Load tools and wrap them for thread safety
try:
    print(f"Attempting to connect to MCP Servers...")
    _raw_mcp_tools = run_async(_init_mcp_client())

    mcp_tools = []
    if _raw_mcp_tools:
        for t in _raw_mcp_tools:

            def create_safe_wrapper(target_tool):
                def safe_invoke(*args, **kwargs):
                    return run_async(target_tool.ainvoke(kwargs))

                return safe_invoke

            safe_tool = StructuredTool.from_function(
                func=create_safe_wrapper(t),
                name=t.name,
                description=t.description,
                args_schema=t.args_schema,
            )
            mcp_tools.append(safe_tool)
        print(f"✅ MCP Tools successfully loaded: {[t.name for t in mcp_tools]}")
    else:
        print("⚠️ Connected to MCP, but no tools were returned.")
except Exception as e:
    print(f"❌ Failed to load MCP tools: {e}")
    mcp_tools = []

# -------------------
# 2. Tools & LLM
# -------------------
search_tool = DuckDuckGoSearchRun(region="us-en")

embeddings = HuggingFaceEndpointEmbeddings(
    model="sentence-transformers/all-MiniLM-L6-v2",
    task="feature-extraction",
    huggingfacehub_api_token=os.getenv("HUGGINGFACEHUB_API_TOKEN"),
)


def _get_retriever(thread_id: Optional[str]):
    if thread_id and thread_id in _THREAD_RETRIEVERS:
        return _THREAD_RETRIEVERS[thread_id]
    return None


def ingest_pdf(
    file_bytes: bytes, thread_id: str, filename: Optional[str] = None
) -> dict:
    if not file_bytes:
        raise ValueError("No bytes received")

    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_file:
        temp_file.write(file_bytes)
        temp_path = temp_file.name

    try:
        loader = PyPDFLoader(temp_path)
        docs = loader.load()
        splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        chunks = splitter.split_documents(docs)
        vector_store = FAISS.from_documents(chunks, embeddings)

        _THREAD_RETRIEVERS[str(thread_id)] = vector_store.as_retriever(
            search_kwargs={"k": 4}
        )
        _THREAD_METADATA[str(thread_id)] = {
            "filename": filename or os.path.basename(temp_path),
            "documents": len(docs),
            "chunks": len(chunks),
        }
        return _THREAD_METADATA[str(thread_id)]
    finally:
        try:
            os.remove(temp_path)
        except:
            pass


@tool
def rag_tool(query: str, thread_id: Optional[str] = None) -> dict:
    """Retrieve info from PDF. thread_id is required."""
    retriever = _get_retriever(thread_id)
    if not retriever:
        return {"error": "No PDF indexed."}
    result = retriever.invoke(query)
    return {"context": [d.page_content for d in result]}


@tool
def get_stock_price(symbol: str) -> dict:
    """Fetch stock price."""
    url = f"https://www.alphavantage.co/query?function=GLOBAL_QUOTE&symbol={symbol}&apikey=C9PE94QUEW9VWGFM"
    return requests.get(url).json()


tools = [search_tool, get_stock_price, *mcp_tools, rag_tool]

# Setup Model
llm_endpoint = HuggingFaceEndpoint(repo_id="deepseek-ai/DeepSeek-V3.2")
model = ChatHuggingFace(llm=llm_endpoint)
llm_with_tools = model.bind_tools(tools)


# -------------------
# 3. State & Graph Definition
# -------------------
class ChatState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]


async def chat_node(state: ChatState, config=None):
    thread_id = config.get("configurable", {}).get("thread_id") if config else None
    sys_msg = SystemMessage(
        content=(
            "You are a helpful assistant. "
            "Use 'search_google_scholar' for research papers. "
            "Use 'rag_tool' for uploaded PDFs. "
            "Use 'add' or 'multiply' for math. "
            f"Current thread ID: `{thread_id}`"
        )
    )
    response = await llm_with_tools.ainvoke(
        [sys_msg, *state["messages"]], config=config
    )
    return {"messages": [response]}


tool_node = ToolNode(tools)

# Define Graph Layout
graph_builder = StateGraph(ChatState)
graph_builder.add_node("chat_node", chat_node)
graph_builder.add_node("tools", tool_node)
graph_builder.add_edge(START, "chat_node")
graph_builder.add_conditional_edges("chat_node", tools_condition)
graph_builder.add_edge("tools", "chat_node")

# -------------------
# 4. Database Access Helpers (Patched)
# -------------------

DB_PATH = "my_chatbot.db"


async def init_title_table(conn):
    """Creates a table for thread titles if it doesn't exist."""
    await conn.execute(
        """
        CREATE TABLE IF NOT EXISTS thread_titles (
            thread_id TEXT PRIMARY KEY,
            title TEXT
        )
        """
    )
    await conn.commit()


@asynccontextmanager
async def _get_patched_saver():
    """
    Creates a connection and patches it to satisfy LangGraph's
    'is_alive' check, then yields the saver.
    """
    conn = await aiosqlite.connect(DB_PATH)

    await init_title_table(conn)

    # --- PATCH START ---
    if not hasattr(conn, "is_alive"):
        conn.is_alive = lambda: True
    # --- PATCH END ---

    saver = AsyncSqliteSaver(conn)
    try:
        await saver.setup()
        yield saver
    finally:
        await conn.close()


@asynccontextmanager
async def get_chatbot_graph():
    """
    Yields a compiled graph with an active database connection.
    Usage: async with get_chatbot_graph() as chatbot: ...
    """
    async with _get_patched_saver() as checkpointer:
        yield graph_builder.compile(checkpointer=checkpointer)


async def _get_thread_history_async(thread_id: str):
    """Async helper to get history."""
    async with _get_patched_saver() as checkpointer:
        temp_graph = graph_builder.compile(checkpointer=checkpointer)
        config = {"configurable": {"thread_id": thread_id}}
        state = await temp_graph.aget_state(config)
        return state.values.get("messages", [])


def get_thread_history(thread_id: str):
    """Sync wrapper for Streamlit to load history."""
    return asyncio.run(_get_thread_history_async(thread_id))


async def _retrieve_all_threads_async():
    """Async helper to get all thread IDs and their titles."""
    async with _get_patched_saver() as checkpointer:
        all_threads = set()
        async for checkpoint in checkpointer.alist(None):
            all_threads.add(checkpoint.config["configurable"]["thread_id"])

        # Now fetch titles for these threads
        conn = checkpointer.conn
        titles = {}
        if all_threads:
            placeholders = ",".join("?" for _ in all_threads)
            async with conn.execute(
                f"SELECT thread_id, title FROM thread_titles WHERE thread_id IN ({placeholders})",
                list(all_threads),
            ) as cursor:
                async for row in cursor:
                    titles[row[0]] = row[1]

        result = []
        for tid in all_threads:
            result.append({"id": tid, "title": titles.get(tid, None)})
        return result


def retrieve_all_threads():
    """Sync wrapper for Streamlit sidebar."""
    return asyncio.run(_retrieve_all_threads_async())


async def _rename_thread_async(thread_id: str, new_title: str):
    async with aiosqlite.connect(DB_PATH) as conn:
        await init_title_table(conn)
        await conn.execute(
            "INSERT OR REPLACE INTO thread_titles (thread_id, title) VALUES (?, ?)",
            (thread_id, new_title),
        )
        await conn.commit()


def rename_thread(thread_id: str, new_title: str):
    asyncio.run(_rename_thread_async(thread_id, new_title))


def thread_document_metadata(thread_id: str) -> dict:
    return _THREAD_METADATA.get(str(thread_id), {})
