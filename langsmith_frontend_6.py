import streamlit as st
from langsmith_backend_6_2_with_traceable import workflow, retrieve_all_threads
from langchain_core.messages import HumanMessage
import uuid


# ------------------ Utility: Generate Unique Thread ID ------------------
def generate_thread_id():
    # Always return string (LangGraph requires string thread_id)
    return str(uuid.uuid4())


# ------------------ Add Thread to Sidebar List ------------------
def add_thread(thread_id):
    if thread_id not in st.session_state["chat_threads"]:
        st.session_state["chat_threads"].append(thread_id)


# ------------------ Reset Chat ------------------
def reset_chat():
    # Save current thread before switching
    add_thread(st.session_state["thread_id"])

    # Create new thread
    st.session_state["thread_id"] = generate_thread_id()

    # Clear UI history
    st.session_state["message_history"] = []

    st.rerun()


# ------------------ Load Conversation From SQLite ------------------
def load_conversation(thread_id):

    state_snapshot = workflow.get_state(
        config={"configurable": {"thread_id": thread_id}}
    )

    # If no previous conversation exists
    if not state_snapshot or not state_snapshot.values:
        return []

    # Safely get messages
    messages = state_snapshot.values.get("messages", [])

    return messages


# ------------------ SESSION INITIALIZATION ------------------

if "message_history" not in st.session_state:
    st.session_state["message_history"] = []

if "thread_id" not in st.session_state:
    st.session_state["thread_id"] = generate_thread_id()

if "chat_threads" not in st.session_state:
    st.session_state["chat_threads"] = retrieve_all_threads()

add_thread(st.session_state["thread_id"])


# ------------------ SIDEBAR ------------------

st.sidebar.title("LangGraph Chatbot")

if st.sidebar.button("New Chat"):
    reset_chat()

st.sidebar.header("My Conversations")

# Show threads in reverse order (latest first)
for thread_id in st.session_state["chat_threads"][::-1]:

    if st.sidebar.button(thread_id):

        st.session_state["thread_id"] = thread_id

        messages = load_conversation(thread_id)

        # Convert LangChain messages → UI format
        temp_messages = []

        for message in messages:

            if isinstance(message, HumanMessage):
                role = "user"
            else:
                role = "assistant"

            temp_messages.append(
                {"role": role, "content": message.content}
            )

        st.session_state["message_history"] = temp_messages
        st.rerun()


# ------------------ DISPLAY OLD MESSAGES ------------------

for message in st.session_state["message_history"]:
    with st.chat_message(message["role"]):
        st.write(message["content"])


# ------------------ HANDLE NEW USER INPUT ------------------

user_input = st.chat_input("Type here")

if user_input:

    # Save user message in UI
    st.session_state["message_history"].append(
        {"role": "user", "content": user_input}
    )

    with st.chat_message("user"):
        st.write(user_input)

    CONFIG = {
        "configurable": {
            "thread_id": st.session_state["thread_id"],#langgrpah memory
            "metadata": st.session_state["thread_id"],#langsmith integration note : langsmith automatically auto inject trace
            "run_name": "chat_run" 
        }
    }

    # Stream assistant response
    with st.chat_message("assistant"):

        def generate_stream():
            for chunk, metadata in workflow.stream(
                {"messages": [HumanMessage(content=user_input)]},
                config=CONFIG,
                stream_mode="messages",
            ):
                if chunk.content:
                    yield chunk.content

        ai_message = st.write_stream(generate_stream)

    # Save assistant message in UI
    st.session_state["message_history"].append(
        {"role": "assistant", "content": ai_message}
    )
