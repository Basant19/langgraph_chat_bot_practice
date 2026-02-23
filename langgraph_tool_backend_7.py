
from langgraph.graph import StateGraph, START
from typing import TypedDict, Annotated, Dict
from langchain_core.messages import BaseMessage, HumanMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition
from langchain_community.tools import DuckDuckGoSearchRun
from langchain_core.tools import tool
from dotenv import load_dotenv
import sqlite3
import requests
import os
from datetime import datetime

# ======================= ENV & CLIENTS ===========================
load_dotenv()
api_key = os.getenv("GOOGLE_API_KEY")

llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    google_api_key=api_key
)

# =========================== TOOLS ===============================
#search_tool = DuckDuckGoSearchRun(region="us-en") inbult but here we are using tavily

@tool
def calculator(first_num: float, second_num: float, operation: str) -> dict:
    """
    Perform a basic arithmetic operation on two numbers.
    Supported operations: add, sub, mul, div
    """
    try:
        if operation == "add":
            result = first_num + second_num
        elif operation == "sub":
            result = first_num - second_num
        elif operation == "mul":
            result = first_num * second_num
        elif operation == "div":
            if second_num == 0:
                return {"error": "Division by zero is not allowed"}
            result = first_num / second_num
        else:
            return {"error": f"Unsupported operation '{operation}'"}
        return {"first_num": first_num, "second_num": second_num, "operation": operation, "result": result}
    except Exception as e:
        return {"error": str(e)}

@tool
def get_stock_price(symbol: str) -> dict:
    """
    Fetch latest stock price for a given symbol via Alpha Vantage.
    """
    _api_key = os.getenv("ALPHAVANTAGE_API_KEY")
    url = f"https://www.alphavantage.co/query?function=GLOBAL_QUOTE&symbol={symbol}&apikey={_api_key}"
    response = requests.get(url, timeout=30)
    return response.json()

@tool
def get_weather(city: str, country: str = "India") -> dict:
    """
    Fetch the current weather for a given city (defaults to India).
    """
    _api_key = os.getenv("WEATHERSTACK_API_KEY")
    query = f"{city},{country}" if country else city
    url = f"http://api.weatherstack.com/current?access_key={_api_key}&query={query}"
    response = requests.get(url, timeout=30)
    return response.json()


@tool
def search_internet(query: str) -> str:
    """
    Search the internet for recent information using Tavily.
    Returns a summarized response suitable for LLM reasoning.
    """
    api_key = os.getenv("TAVILY_API_KEY")
    if not api_key:
        return "Tavily API key not found."

    url = "https://api.tavily.com/search"

    payload = {
        "api_key": api_key,
        "query": query,
        "search_depth": "advanced",  # better results than basic
        "include_answer": True,
        "max_results": 3
    }

    try:
        response = requests.post(url, json=payload, timeout=30)
        response.raise_for_status()
        data = response.json()

        # If Tavily provides summarized answer
        if "answer" in data and data["answer"]:
            return data["answer"]

        # Else fallback to top results
        results = data.get("results", [])
        if not results:
            return "No relevant search results found."

        combined = "\n\n".join(
            f"Source: {r.get('url', 'N/A')}\nContent: {r.get('content', '')[:500]}"
            for r in results
        )

        return combined

    except requests.exceptions.RequestException as e:
        return f"Error calling Tavily API: {str(e)}"


tools = [ get_stock_price, get_weather, calculator,search_internet]
llm_with_tools = llm.bind_tools(tools)  # allow LLM to decide to call tools

# =========================== STATE ===============================
class ChatState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]

def chat_node(state: ChatState):
    """
    LLM node that may answer directly or request a tool call.
    If a tool returns raw JSON, we ask LLM to produce a readable answer.
    """
    messages = state["messages"]
    response = llm_with_tools.invoke(messages)

    # Some tool integrations may return dicts; wrap into human text
    if isinstance(response, dict):
        formatted = llm.invoke([
            *messages,
            HumanMessage(content=f"Format this tool result into a helpful answer: {response}")
        ])
        return {"messages": [formatted]}

    return {"messages": [response]}

tool_node = ToolNode(tools)

# ======================== CHECKPOINTER ===========================
conn = sqlite3.connect(database="chatbot.db", check_same_thread=False)
checkpointer = SqliteSaver(conn=conn)

# =========================== GRAPH ===============================
graph = StateGraph(ChatState)
graph.add_node("chat_node", chat_node)
graph.add_node("tools", tool_node)
graph.add_edge(START, "chat_node")
graph.add_conditional_edges("chat_node", tools_condition)
graph.add_edge("tools", "chat_node")

chatbot = graph.compile(checkpointer=checkpointer)

# ===================== CHAT METADATA (SQLite) ====================
# Persist thread names + soft delete flag so they survive restarts.

def init_metadata_table():
    conn.execute("""
    CREATE TABLE IF NOT EXISTS chat_metadata (
        thread_id TEXT PRIMARY KEY,
        chat_name TEXT NOT NULL,
        deleted INTEGER NOT NULL DEFAULT 0,
        created_at TEXT NOT NULL DEFAULT (datetime('now')),
        updated_at TEXT NOT NULL DEFAULT (datetime('now'))
    )
    """)
    conn.commit()

init_metadata_table()

def _upsert_name(thread_id: str, chat_name: str):
    now = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S")
    conn.execute("""
    INSERT INTO chat_metadata (thread_id, chat_name, deleted, created_at, updated_at)
    VALUES (?, ?, 0, COALESCE((SELECT created_at FROM chat_metadata WHERE thread_id = ?), ?), ?)
    ON CONFLICT(thread_id) DO UPDATE SET
        chat_name=excluded.chat_name,
        deleted=0,
        updated_at=excluded.updated_at
    """, (thread_id, chat_name, thread_id, now, now))
    conn.commit()

def save_chat_name(thread_id: str, chat_name: str):
    """Create/update chat name (also clears 'deleted' flag if it was deleted)."""
    _upsert_name(str(thread_id), chat_name)

def get_chat_name(thread_id: str) -> str | None:
    cur = conn.execute("SELECT chat_name FROM chat_metadata WHERE thread_id = ? AND deleted = 0", (str(thread_id),))
    row = cur.fetchone()
    return row[0] if row else None

def get_all_chats() -> Dict[str, str]:
    """
    Return all chats (thread_id -> chat_name) that are NOT deleted.
    - If a thread exists in LangGraph checkpoints but has no metadata yet,
      it will be given a default name.
    """
    # Load existing names
    cur = conn.execute("SELECT thread_id, chat_name FROM chat_metadata WHERE deleted = 0")
    meta = {tid: name for tid, name in cur.fetchall()}

    # Add any threads known to the checkpointer but missing in metadata
    for checkpoint in checkpointer.list(None):
        tid = str(checkpoint.config["configurable"]["thread_id"])
        if tid not in meta:
            # generate a deterministic default name
            meta[tid] = f"Chat {len(meta) + 1}"
            _upsert_name(tid, meta[tid])

    return meta

def rename_chat(thread_id: str, new_name: str):
    save_chat_name(str(thread_id), new_name)

def delete_chat(thread_id: str, hard: bool = False):
    """
    Soft delete by default: mark deleted in chat_metadata so it disappears from UI.
    Optional hard purge tries to remove checkpointer rows (best-effort).
    """
    tid = str(thread_id)
    conn.execute("UPDATE chat_metadata SET deleted = 1, updated_at = datetime('now') WHERE thread_id = ?", (tid,))
    conn.commit()

    if hard:
        # Best-effort purge of LangGraph SqliteSaver data for this thread.
        # Schema can vary by version; we defensively check if tables exist.
        def table_exists(name: str) -> bool:
            c = conn.execute("SELECT name FROM sqlite_master WHERE type='table' AND name=?", (name,))
            return c.fetchone() is not None

        # Common table names used by SqliteSaver across versions
        candidate_tables = [
            "checkpoints",
            "checkpoint_blobs",
            "checkpoint_writes",
            "writes",
            "writes_blobs",
        ]

        for table in candidate_tables:
            if table_exists(table):
                try:
                    conn.execute(f"DELETE FROM {table} WHERE thread_id = ?", (tid,))
                except Exception:
                    # Some tables may use a nested JSON or different schema; ignore if not applicable
                    pass
        conn.commit()

# Back-compat helper (return
