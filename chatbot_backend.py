import sys
import json
import re
import shutil
import asyncio
import threading
import requests
import os
import tempfile
import aiosqlite
from dotenv import load_dotenv
from typing import Annotated, Dict, Optional, TypedDict
from contextlib import asynccontextmanager

from langgraph.graph import StateGraph, START
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
from langchain_core.documents import Document
from langchain_core.messages import SystemMessage, BaseMessage
from langchain_core.tools import tool, StructuredTool
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_opendataloader_pdf import OpenDataLoaderPDFLoader
from langchain_groq import ChatGroq
from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain_text_splitters import (
    MarkdownHeaderTextSplitter,
    RecursiveCharacterTextSplitter,
)
from langchain_community.vectorstores import FAISS

load_dotenv()

# -------------------
# 0. Async Runner Setup
# -------------------
_ASYNC_LOOP = asyncio.new_event_loop()
_ASYNC_THREAD = threading.Thread(target=_ASYNC_LOOP.run_forever, daemon=True)
_ASYNC_THREAD.start()

_THREAD_VECTORSTORES: Dict[str, FAISS] = {}
_THREAD_METADATA: Dict[str, dict] = {}
_VECTORSTORE_ROOT = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "vectorstores"
)


def run_async(coro):
    """Helper to run async code on the background thread."""
    return asyncio.run_coroutine_threadsafe(coro, _ASYNC_LOOP).result()


# -------------------
# 1. MCP Client Setup (Updated)
# -------------------
_CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))

client = MultiServerMCPClient(
    {
        # Existing Arithmetic Server
        "arithmetic": {
            "transport": "stdio",
            "command": sys.executable,
            "args": [os.path.join(_CURRENT_DIR, "mcp_server.py")],
        },
        # NEW: Google Scholar Server
        "scholar": {
            "transport": "stdio",
            "command": sys.executable,
            "args": [os.path.join(_CURRENT_DIR, "scholar_server.py")],
        },
    }
)


async def _init_mcp_client():
    # This fetches tools from ALL connected servers (arithmetic + scholar)
    return await client.get_tools()


mcp_load_error = None

# Load tools and wrap them for thread safety
try:
    print("Attempting to connect to MCP Servers...")
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
        print(f"[SUCCESS] MCP Tools successfully loaded: {[t.name for t in mcp_tools]}")
    else:
        print("[WARNING] Connected to MCP, but no tools were returned.")
        mcp_load_error = "Connected to MCP, but no tools were returned."
except Exception as e:
    print(f"[ERROR] Failed to load MCP tools: {e}")
    mcp_load_error = str(e)
    mcp_tools = []


def get_mcp_error():
    return mcp_load_error

@tool("duckduckgo_search")
def search_tool(query: str) -> str:
    """Search the web for current events, news, and general real-time information."""
    # 1. Try DuckDuckGo text search
    try:
        from ddgs import DDGS
        results = list(DDGS(timeout=5).text(query, max_results=3))
        if results:
            return "\n\n".join(
                f"[{r.get('title')}]({r.get('href')}): {r.get('body')}"
                for r in results
            )
    except Exception:
        pass

    # 2. Try DuckDuckGo news search
    try:
        from ddgs import DDGS
        results = list(DDGS(timeout=5).news(query, max_results=3))
        if results:
            return "\n\n".join(
                f"[{r.get('title')}]({r.get('url')}): {r.get('body')}"
                for r in results
            )
    except Exception:
        pass

    # 3. Fallback to SerpApi (Google Search) if DuckDuckGo times out or is blocked
    serp_key = os.getenv("SERPAPI_API_KEY")
    if serp_key:
        try:
            from serpapi import GoogleSearch
            params = {"q": query, "api_key": serp_key, "num": 3}
            search = GoogleSearch(params)
            organic = search.get_dict().get("organic_results", [])
            if organic:
                return "\n\n".join(
                    f"[{item.get('title')}]({item.get('link')}): {item.get('snippet', '')}"
                    for item in organic
                )
        except Exception as e:
            return f"Search fallback error: {e}"

    return "No search results could be retrieved at this time."

embeddings = GoogleGenerativeAIEmbeddings(
    model="models/gemini-embedding-001",
    google_api_key=os.getenv("GOOGLE_API_KEY"),
)


def _thread_store_dir(thread_id: str) -> str:
    safe_id = re.sub(r"[^a-zA-Z0-9_-]", "_", str(thread_id))
    return os.path.join(_VECTORSTORE_ROOT, safe_id)


def _get_vector_store(thread_id: Optional[str]) -> Optional[FAISS]:
    if not thread_id:
        return None
    tid = str(thread_id)
    if tid in _THREAD_VECTORSTORES:
        return _THREAD_VECTORSTORES[tid]

    store_dir = _thread_store_dir(tid)
    index_path = os.path.join(store_dir, "index.faiss")
    if os.path.exists(index_path):
        vector_store = FAISS.load_local(
            store_dir, embeddings, allow_dangerous_deserialization=True
        )
        _THREAD_VECTORSTORES[tid] = vector_store
        return vector_store
    return None


def _get_retriever(thread_id: Optional[str]):
    vector_store = _get_vector_store(thread_id)
    if vector_store is not None:
        return vector_store.as_retriever(search_kwargs={"k": 4})
    return None


def _split_markdown_documents(docs: list[Document]) -> list[Document]:
    """Split Markdown pages by headings first, then into size-bounded chunks."""
    header_splitter = MarkdownHeaderTextSplitter(
        headers_to_split_on=[
            ("#", "h1"),
            ("##", "h2"),
            ("###", "h3"),
        ],
        strip_headers=False,
    )
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)

    header_docs: list[Document] = []
    for doc in docs:
        if not doc.page_content or not doc.page_content.strip():
            continue
        sections = header_splitter.split_text(doc.page_content)
        if not sections:
            header_docs.append(doc)
            continue
        for section in sections:
            merged_meta = {**doc.metadata, **section.metadata}
            header_docs.append(
                Document(page_content=section.page_content, metadata=merged_meta)
            )

    return text_splitter.split_documents(header_docs)


def ingest_pdf(
    file_bytes: bytes, thread_id: str, filename: Optional[str] = None
) -> dict:
    if not file_bytes:
        raise ValueError("No bytes received")

    tid = str(thread_id)
    display_name = filename or "uploaded.pdf"

    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_file:
        temp_file.write(file_bytes)
        temp_path = temp_file.name

    try:
        loader = OpenDataLoaderPDFLoader(
            file_path=temp_path,
            format="markdown",
            split_pages=True,
            table_method="cluster",
            reading_order="xycut",
            quiet=True,
        )
        docs = loader.load()
        if not docs:
            raise ValueError(
                "No extractable text found in this PDF (it may be scanned, image-only, or encrypted)."
            )

        for doc in docs:
            doc.metadata["source"] = display_name

        chunks = _split_markdown_documents(docs)
        if not chunks:
            raise ValueError(
                "No extractable text found in this PDF (it may be scanned, image-only, or encrypted)."
            )

        existing_store = _get_vector_store(tid)
        if existing_store is not None:
            existing_store.add_documents(chunks)
            vector_store = existing_store
        else:
            vector_store = FAISS.from_documents(chunks, embeddings)
            _THREAD_VECTORSTORES[tid] = vector_store

        store_dir = _thread_store_dir(tid)
        os.makedirs(store_dir, exist_ok=True)
        vector_store.save_local(store_dir)

        prev_meta = thread_document_metadata(tid)
        files_list = list(prev_meta.get("files", []))
        file_entry = {
            "filename": display_name,
            "documents": len(docs),
            "chunks": len(chunks),
        }
        files_list = [f for f in files_list if f.get("filename") != display_name]
        files_list.append(file_entry)

        meta = {
            "filename": ", ".join(f["filename"] for f in files_list),
            "documents": sum(f["documents"] for f in files_list),
            "chunks": sum(f["chunks"] for f in files_list),
            "files": files_list,
            "latest_file": file_entry,
        }
        _THREAD_METADATA[tid] = meta
        with open(os.path.join(store_dir, "metadata.json"), "w", encoding="utf-8") as f:
            json.dump(meta, f, ensure_ascii=False, indent=2)

        return meta
    finally:
        try:
            os.remove(temp_path)
        except OSError:
            pass


@tool
def rag_tool(query: str, thread_id: Optional[str] = None) -> dict:
    """Retrieve relevant passages and page metadata from uploaded PDFs. thread_id is required."""
    retriever = _get_retriever(thread_id)
    if not retriever:
        return {"error": "No PDF indexed."}
    result = retriever.invoke(query)
    return {
        "context": [
            {
                "source": d.metadata.get("source"),
                "page": d.metadata.get("page"),
                "content": d.page_content,
            }
            for d in result
        ]
    }


@tool
def get_stock_price(symbol: str) -> dict:
    """Fetch stock price."""
    api_key = os.getenv("ALPHAVANTAGE_API_KEY")
    if not api_key:
        return {"error": "ALPHAVANTAGE_API_KEY is not configured in .env."}
    try:
        resp = requests.get(
            "https://www.alphavantage.co/query",
            params={
                "function": "GLOBAL_QUOTE",
                "symbol": symbol,
                "apikey": api_key,
            },
            timeout=10,
        )
        resp.raise_for_status()
        return resp.json()
    except requests.RequestException as e:
        return {"error": f"Stock API request failed: {e}"}


tools = [search_tool, get_stock_price, *mcp_tools, rag_tool]

# Setup Groq Model
model = ChatGroq(
    model="openai/gpt-oss-120b",
    api_key=os.getenv("GROQ_API_KEY"),
    temperature=0.2,
)

# Bind existing tools to the model
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
            "Use 'duckduckgo_search' for current web searches, real-time facts, and news. "
            "Use 'get_stock_price' for stock market prices. "
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


tool_node = ToolNode(tools, handle_tool_errors=True)

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
    return run_async(_get_thread_history_async(thread_id))


async def _retrieve_all_threads_async():
    """Async helper to get all thread IDs and their titles."""
    async with _get_patched_saver() as checkpointer:
        all_threads = set()
        
        # 1. Get threads with history (checkpoints)
        try:
            async for checkpoint in checkpointer.alist(None):
                all_threads.add(checkpoint.config["configurable"]["thread_id"])
        except Exception:
            pass # Handle case where checkpoint table might be empty/init issues

        conn = checkpointer.conn
        
        # 2. Get threads that have explicitly named titles (even if no history yet)
        # We need to ensure the table exists or handle error if simpler
        try:
            async with conn.execute("SELECT thread_id FROM thread_titles") as cursor:
                async for row in cursor:
                    all_threads.add(row[0])
        except Exception:
            pass # Table might not exist yet if no renames happened

        # 3. Fetch titles for these threads
        titles = {}
        if all_threads:
            placeholders = ",".join("?" for _ in all_threads)
            query = f"SELECT thread_id, title FROM thread_titles WHERE thread_id IN ({placeholders})"
            try:
                async with conn.execute(query, list(all_threads)) as cursor:
                    async for row in cursor:
                        titles[row[0]] = row[1]
            except Exception:
                pass

        result = []
        for tid in all_threads:
            result.append({"id": tid, "title": titles.get(tid, None)})
        return result


def retrieve_all_threads():
    """Sync wrapper for Streamlit sidebar."""
    return run_async(_retrieve_all_threads_async())


async def _rename_thread_async(thread_id: str, new_title: str):
    async with aiosqlite.connect(DB_PATH) as conn:
        await init_title_table(conn)
        await conn.execute(
            "INSERT OR REPLACE INTO thread_titles (thread_id, title) VALUES (?, ?)",
            (thread_id, new_title),
        )
        await conn.commit()


def rename_thread(thread_id: str, new_title: str):
    run_async(_rename_thread_async(thread_id, new_title))


async def _delete_thread_async(thread_id: str):
    """Delete a thread's checkpoints, writes, title, and persisted RAG artifacts."""
    tid = str(thread_id)
    async with _get_patched_saver() as checkpointer:
        await checkpointer.adelete_thread(tid)
        await checkpointer.conn.execute(
            "DELETE FROM thread_titles WHERE thread_id = ?", (tid,)
        )
        await checkpointer.conn.commit()

    _THREAD_VECTORSTORES.pop(tid, None)
    _THREAD_METADATA.pop(tid, None)
    store_dir = _thread_store_dir(tid)
    if os.path.exists(store_dir):
        shutil.rmtree(store_dir, ignore_errors=True)


def delete_thread(thread_id: str):
    """Sync wrapper for Streamlit to delete a conversation thread."""
    run_async(_delete_thread_async(thread_id))


def thread_document_metadata(thread_id: str) -> dict:
    tid = str(thread_id)
    if tid in _THREAD_METADATA:
        return _THREAD_METADATA[tid]

    meta_path = os.path.join(_thread_store_dir(tid), "metadata.json")
    if os.path.exists(meta_path):
        try:
            with open(meta_path, "r", encoding="utf-8") as f:
                meta = json.load(f)
            _THREAD_METADATA[tid] = meta
            return meta
        except (OSError, json.JSONDecodeError):
            pass
    return {}
