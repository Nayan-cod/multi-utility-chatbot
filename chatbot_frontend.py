import uuid
import asyncio
import streamlit as st
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

# Import from your backend
from chatbot_backend import (
    get_chatbot_graph,
    ingest_pdf,
    retrieve_all_threads,
    thread_document_metadata,
    get_thread_history,
    rename_thread,  # <--- New Import
)


# =========================== Utilities ===========================
def generate_thread_id():
    return str(uuid.uuid4())


def reset_chat():
    thread_id = generate_thread_id()
    st.session_state["thread_id"] = thread_id
    add_thread({"id": thread_id, "title": None})
    st.session_state["message_history"] = []


def add_thread(thread_data):
    # Check if thread exists in list based on ID
    exists = any(t["id"] == thread_data["id"] for t in st.session_state["chat_threads"])
    if not exists:
        st.session_state["chat_threads"].append(thread_data)


def load_conversation(thread_id):
    return get_thread_history(thread_id)


# ======================= Session Initialization ===================
if "message_history" not in st.session_state:
    st.session_state["message_history"] = []

if "thread_id" not in st.session_state:
    st.session_state["thread_id"] = generate_thread_id()

# Always reload threads on refresh to get latest titles
st.session_state["chat_threads"] = retrieve_all_threads()

if "ingested_docs" not in st.session_state:
    st.session_state["ingested_docs"] = {}

# Ensure current thread is tracked
current_thread_exists = any(
    t["id"] == st.session_state["thread_id"] for t in st.session_state["chat_threads"]
)
if not current_thread_exists:
    add_thread({"id": st.session_state["thread_id"], "title": None})

thread_key = str(st.session_state["thread_id"])
thread_docs = st.session_state["ingested_docs"].setdefault(thread_key, {})

# Sort threads: You might want to sort by timestamp if available, but for now reverse ID is ok
threads = st.session_state["chat_threads"][::-1]
selected_thread = None

# ============================ Sidebar ============================
st.sidebar.title("LangGraph Chatbot")

# --- Current Chat Settings ---
st.sidebar.caption(f"Current ID: `{thread_key[:8]}...`")

# Rename Input
new_title_input = st.sidebar.text_input(
    "Rename Chat", placeholder="Enter new name", key="rename_input"
)
if st.sidebar.button("Update Title"):
    if new_title_input:
        rename_thread(thread_key, new_title_input)
        st.success("Renamed!")
        st.rerun()  # Refresh to show new name in list

if st.sidebar.button("New Chat", use_container_width=True):
    reset_chat()
    st.rerun()

st.sidebar.divider()

# Document Info Display
if thread_docs:
    latest_doc = list(thread_docs.values())[-1]
    st.sidebar.success(
        f"📄 **Active PDF:**\n{latest_doc.get('filename')}\n"
        f"({latest_doc.get('documents')} pages)"
    )
else:
    st.sidebar.info("No PDF indexed yet.")

# File Uploader
uploaded_pdf = st.sidebar.file_uploader("Upload a PDF for this chat", type=["pdf"])
if uploaded_pdf:
    if uploaded_pdf.name in thread_docs:
        st.sidebar.info(f"✅ `{uploaded_pdf.name}` loaded.")
    else:
        with st.sidebar.status("Indexing PDF…", expanded=True) as status_box:
            summary = ingest_pdf(
                uploaded_pdf.getvalue(),
                thread_id=thread_key,
                filename=uploaded_pdf.name,
            )
            thread_docs[uploaded_pdf.name] = summary
            status_box.update(label="✅ PDF indexed", state="complete", expanded=False)

st.sidebar.divider()
st.sidebar.subheader("Past Conversations")

if not threads:
    st.sidebar.caption("No past conversations.")
else:
    for t_data in threads:
        t_id = t_data["id"]
        t_title = t_data["title"]

        # Display logic: Custom Title OR "Thread [first 8 chars]"
        label = t_title if t_title else f"Thread {str(t_id)[:8]}..."

        if st.sidebar.button(
            label, key=f"side-thread-{t_id}", use_container_width=True
        ):
            selected_thread = t_id

# ============================ Main Layout ========================
# Helper to find current title for Main Title display
current_title_obj = next((t for t in threads if t["id"] == thread_key), None)
main_title = (
    current_title_obj["title"]
    if current_title_obj and current_title_obj["title"]
    else "Multi Utility Chatbot (Async)"
)

st.title(main_title)

# 1. Display Chat History
for message in st.session_state["message_history"]:
    role = message["role"]
    with st.chat_message(role):
        st.markdown(message["content"])

# 2. Handle User Input
user_input = st.chat_input("Ask about your document or use tools")

if user_input:
    st.session_state["message_history"].append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    # 3. Define Async Generator Logic
    async def process_chat_stream():
        """
        Runs the chatbot.astream() and updates Streamlit UI placeholders in real-time.
        """
        config = {
            "configurable": {"thread_id": thread_key},
            "metadata": {"thread_id": thread_key},
            "run_name": "chat_turn",
        }

        with st.chat_message("assistant"):
            status_placeholder = st.empty()
            message_placeholder = st.empty()
            full_response = ""
            tool_status = None

            # --- Use Context Manager for DB Connection ---
            async with get_chatbot_graph() as chatbot:
                async for chunk, _ in chatbot.astream(
                    {"messages": [HumanMessage(content=user_input)]},
                    config=config,
                    stream_mode="messages",
                ):
                    if isinstance(chunk, AIMessage) and chunk.tool_calls:
                        tool_names = [tc["name"] for tc in chunk.tool_calls]
                        if tool_status is None:
                            tool_status = status_placeholder.status(
                                f"⚙️ Running tools: {', '.join(tool_names)}...",
                                expanded=True,
                            )
                        else:
                            tool_status.write(f"Calling: {', '.join(tool_names)}...")

                    elif isinstance(chunk, ToolMessage):
                        if tool_status:
                            tool_status.write(f"Result: {str(chunk.content)[:100]}...")

                    elif isinstance(chunk, AIMessage) and chunk.content:
                        if tool_status:
                            tool_status.update(
                                label="✅ Tools finished",
                                state="complete",
                                expanded=False,
                            )
                            tool_status = None

                        full_response += chunk.content
                        message_placeholder.markdown(full_response + "▌")

            message_placeholder.markdown(full_response)
            if tool_status:
                tool_status.update(label="✅ Done", state="complete", expanded=False)

            return full_response

    # 4. Run the Async Logic Synchronously
    try:
        final_text = asyncio.run(process_chat_stream())

        st.session_state["message_history"].append(
            {"role": "assistant", "content": final_text}
        )

        doc_meta = thread_document_metadata(thread_key)
        if doc_meta:
            st.caption(
                f"Source: {doc_meta.get('filename')} "
                f"(chunks: {doc_meta.get('chunks')})"
            )

    except Exception as e:
        st.error(f"An error occurred: {e}")


st.divider()

# ============================ Thread Loading Logic ========================
if selected_thread:
    st.session_state["thread_id"] = selected_thread
    raw_messages = load_conversation(selected_thread)

    clean_history = []
    for msg in raw_messages:
        if isinstance(msg, HumanMessage):
            clean_history.append({"role": "user", "content": msg.content})
        elif isinstance(msg, AIMessage) and msg.content:
            clean_history.append({"role": "assistant", "content": msg.content})

    st.session_state["message_history"] = clean_history
    st.session_state["ingested_docs"].setdefault(str(selected_thread), {})
    st.rerun()
