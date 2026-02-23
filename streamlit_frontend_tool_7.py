import streamlit as st
from langgraph_tool_backend_7 import (
    chatbot,
    get_all_chats,
    save_chat_name,
    rename_chat,
    delete_chat,
)
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage
import uuid

# =========================== Utilities ===========================
def generate_thread_id() -> str:
    return str(uuid.uuid4())

def reset_chat():
    thread_id = generate_thread_id()
    default_name = f"Chat {len(st.session_state['chat_threads']) + 1}"
    add_thread(thread_id, default_name)
    save_chat_name(thread_id, default_name)
    st.session_state["thread_id"] = thread_id
    st.session_state["message_history"] = []

def add_thread(thread_id: str, default_name: str | None = None):
    if thread_id not in st.session_state["chat_threads"]:
        name = default_name or f"Chat {len(st.session_state['chat_threads']) + 1}"
        st.session_state["chat_threads"][thread_id] = name
        save_chat_name(thread_id, name)

def load_conversation(thread_id: str):
    state = chatbot.get_state(config={"configurable": {"thread_id": thread_id}})
    return state.values.get("messages", [])

def ensure_current_thread_valid():
    if not st.session_state["chat_threads"]:
        # If nothing exists, create a fresh one
        reset_chat()
        return
    if st.session_state["thread_id"] not in st.session_state["chat_threads"]:
        # Pick any existing one
        st.session_state["thread_id"] = next(iter(st.session_state["chat_threads"].keys()))

# ======================= Session Initialization ===================
if "message_history" not in st.session_state:
    st.session_state["message_history"] = []

if "chat_threads" not in st.session_state:
    # Load persisted chats from DB (thread_id -> name)
    st.session_state["chat_threads"] = get_all_chats()

if "thread_id" not in st.session_state:
    # Choose first existing, else create
    if st.session_state["chat_threads"]:
        st.session_state["thread_id"] = next(iter(st.session_state["chat_threads"].keys()))
    else:
        st.session_state["thread_id"] = generate_thread_id()
        default_name = "Chat 1"
        st.session_state["chat_threads"][st.session_state["thread_id"]] = default_name
        save_chat_name(st.session_state["thread_id"], default_name)

ensure_current_thread_valid()

# ============================ Sidebar ============================
st.sidebar.title("LangGraph Chatbot")

if st.sidebar.button("➕ New Chat"):
    reset_chat()

st.sidebar.header("My Conversations")

# List with per-item delete buttons
with st.sidebar.container():
    # Show newest first in the list
    items = list(st.session_state["chat_threads"].items())[::-1]
    for thread_id, chat_name in items:
        col1, col2 = st.columns([0.78, 0.22], gap="small")
        with col1:
            if st.button(chat_name, key=f"sel_{thread_id}"):
                st.session_state["thread_id"] = thread_id
                messages = load_conversation(thread_id)
                temp_messages = []
                for msg in messages:
                    role = "user" if isinstance(msg, HumanMessage) else "assistant"
                    temp_messages.append({"role": role, "content": msg.content})
                st.session_state["message_history"] = temp_messages
        with col2:
            if st.button("🗑", key=f"del_{thread_id}", help="Delete chat"):
                # Soft delete (remove from UI + DB). Pass hard=True to try DB purge.
                delete_chat(thread_id, hard=False)
                # Remove from session, and if deleting current, pick another
                st.session_state["chat_threads"].pop(thread_id, None)
                if st.session_state["thread_id"] == thread_id:
                    if st.session_state["chat_threads"]:
                        st.session_state["thread_id"] = next(iter(st.session_state["chat_threads"].keys()))
                    else:
                        reset_chat()
                st.rerun()

# Rename functionality
st.sidebar.subheader("Rename Chat")
current_thread_id = st.session_state["thread_id"]
current_name = st.session_state["chat_threads"].get(current_thread_id, "Untitled Chat")

new_name = st.sidebar.text_input(
    "Enter new name", value=current_name, key="rename_input_tool"
)

if st.sidebar.button("💾 Save Name"):
    st.session_state["chat_threads"][current_thread_id] = new_name
    rename_chat(current_thread_id, new_name)
    st.sidebar.success("Saved!")
    # No rerun needed; UI will reflect from session state.

# ============================ Main UI ============================
# Render history
for message in st.session_state["message_history"]:
    with st.chat_message(message["role"]):
        st.text(message["content"])

user_input = st.chat_input("Type here")

if user_input:
    # Auto-update name on first turn in a brand-new chat
    if len(st.session_state["message_history"]) == 0:
        truncated_name = user_input.strip()[:30]
        st.session_state["chat_threads"][current_thread_id] = truncated_name
        rename_chat(current_thread_id, truncated_name)

    # Show user's message
    st.session_state["message_history"].append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.text(user_input)

    CONFIG = {
        "configurable": {"thread_id": current_thread_id},
        "metadata": {"thread_id": current_thread_id},
        "run_name": "chat_turn",
    }

    with st.chat_message("assistant"):
        status_holder = {"box": None}

        def ai_only_stream():
            for message_chunk, metadata in chatbot.stream(
                {"messages": [HumanMessage(content=user_input)]},
                config=CONFIG,
                stream_mode="messages",
            ):
                if isinstance(message_chunk, ToolMessage):
                    tool_name = getattr(message_chunk, "name", "tool")
                    if status_holder["box"] is None:
                        status_holder["box"] = st.status(
                            f"🔧 Using `{tool_name}` …", expanded=True
                        )
                    else:
                        status_holder["box"].update(
                            label=f"🔧 Using `{tool_name}` …",
                            state="running",
                            expanded=True,
                        )
                if isinstance(message_chunk, AIMessage):
                    yield message_chunk.content

        ai_message = st.write_stream(ai_only_stream())

        if status_holder["box"] is not None:
            status_holder["box"].update(
                label="✅ Tool finished", state="complete", expanded=False
            )

    st.session_state["message_history"].append(
        {"role": "assistant", "content": ai_message}
    )
