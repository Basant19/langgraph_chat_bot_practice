from __future__ import annotations

import os
import sqlite3
import tempfile
from typing import Annotated, Any, Dict, Optional, TypedDict

from dotenv import load_dotenv
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader

from langchain_community.vectorstores import FAISS
from langchain_core.messages import BaseMessage, SystemMessage
from langchain_core.tools import tool
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain.chat_models import init_chat_model
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph import START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition
import requests

load_dotenv()
CURRENT_THREAD_ID = None
# -------------------
# 1. LLM + embeddings (Gemini)
# -------------------

api_key = os.getenv("GOOGLE_API_KEY")
ALPHAVANTAGE_API_KEY = os.getenv("ALPHAVANTAGE_API_KEY")
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")
WEATHERSTACK_API_KEY = os.getenv("WEATHERSTACK_API_KEY")

llm = init_chat_model(
    "google_genai:gemini-2.5-flash-lite",
    api_key=api_key
)

embeddings = GoogleGenerativeAIEmbeddings(
    model="models/gemini-embedding-001",
    google_api_key=api_key
)

# -------------------
# 2. PDF retriever store (per thread)
# -------------------

_THREAD_RETRIEVERS: Dict[str, Any] = {}
_THREAD_METADATA: Dict[str, dict] = {}

def _get_retriever(thread_id: Optional[str]):
    if thread_id and thread_id in _THREAD_RETRIEVERS:
        return _THREAD_RETRIEVERS[thread_id]
    return None


def ingest_pdf(file_bytes: bytes, thread_id: str, filename: Optional[str] = None) -> dict:
    if not file_bytes:
        raise ValueError("No bytes received for ingestion.")

    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_file:
        temp_file.write(file_bytes)
        temp_path = temp_file.name

    try:
        loader = PyPDFLoader(temp_path)
        docs = loader.load()

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200
        )
        chunks = splitter.split_documents(docs)

        vector_store = FAISS.from_documents(chunks, embeddings)
        retriever = vector_store.as_retriever(
            search_type="similarity",
            search_kwargs={"k": 4}
        )

        _THREAD_RETRIEVERS[str(thread_id)] = retriever
        _THREAD_METADATA[str(thread_id)] = {
            "filename": filename,
            "documents": len(docs),
            "chunks": len(chunks),
        }

        return _THREAD_METADATA[str(thread_id)]

    finally:
        try:
            os.remove(temp_path)
        except OSError:
            pass

# -------------------
# 3. Tools
# -------------------

# ---------- Tavily Search Tool ----------
@tool
def tavily_search(query: str) -> dict:
    """
    Search the web using Tavily API.
    """
    url = "https://api.tavily.com/search"

    payload = {
        "api_key": TAVILY_API_KEY,
        "query": query,
        "search_depth": "basic"
    }

    response = requests.post(url, json=payload)
    return response.json()


# ---------- Stock Price Tool ----------
@tool
def get_stock_price(symbol: str) -> dict:
    """
    Fetch latest stock price using Alpha Vantage API.
    """
    url = (
        "https://www.alphavantage.co/query"
        f"?function=GLOBAL_QUOTE"
        f"&symbol={symbol}"
        f"&apikey={ALPHAVANTAGE_API_KEY}"
    )

    r = requests.get(url)
    return r.json()


# ---------- Weather Tool ----------
@tool
def get_weather(city: str) -> dict:
    """
    Get current weather using Weatherstack API.
    """
    url = (
        "http://api.weatherstack.com/current"
        f"?access_key={WEATHERSTACK_API_KEY}"
        f"&query={city}"
    )

    r = requests.get(url)
    return r.json()


# ---------- Calculator ----------
@tool
def calculator(first_num: float, second_num: float, operation: str) -> dict:
    """
    Perform basic arithmetic operations.
    Supported operations: add, sub, mul, div.
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
                return {"error": "Division by zero not allowed"}
            result = first_num / second_num
        else:
            return {"error": "Unsupported operation"}

        return {"result": result}

    except Exception as e:
        return {"error": str(e)}
    
# ---------- RAG Tool ----------
@tool
def rag_tool(query: str) -> dict:
    """
    Retrieve relevant information from the uploaded PDF document 
    associated with the current chat thread.
        Use this tool when the user asks questions about:
    - Uploaded PDF content
    - Document summaries
    - Information contained inside the indexed file
        Returns:
    - query: The original question
    - context: Retrieved relevant document chunks
    - source_file: Name of the uploaded document
   
    """

    # thread_id will be injected via config
    from langgraph_rag_backend_8 import CURRENT_THREAD_ID

    retriever = _get_retriever(CURRENT_THREAD_ID)

    if retriever is None:
        return {"error": "No document uploaded for this chat."}

    result = retriever.invoke(query)
    context = [doc.page_content for doc in result]

    return {
        "query": query,
        "context": context,
        "source_file": _THREAD_METADATA.get(str(CURRENT_THREAD_ID), {}).get("filename"),
    }

tools = [
    tavily_search,
    get_stock_price,
    get_weather,
    calculator,
    rag_tool
]

llm_with_tools = llm.bind_tools(tools)


# -------------------
# 4. State
# -------------------

class ChatState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]


# -------------------
# 5. Nodes
# -------------------
def chat_node(state: ChatState, config=None):

    global CURRENT_THREAD_ID

    thread_id = None
    if config:
        thread_id = config.get("configurable", {}).get("thread_id")

    CURRENT_THREAD_ID = str(thread_id)

    system_message = SystemMessage(
    content=(
        "You are a helpful assistant. "
        "For PDF questions, call rag_tool with thread_id. "
        "For web search use tavily_search. "
        "For stock price use get_stock_price. "
        "For weather use get_weather. "
        "For math use calculator."
    )
)

    messages = [system_message, *state["messages"]]

    response = llm_with_tools.invoke(messages, config=config)

    return {"messages": [response]}


tool_node = ToolNode(tools)
# -------------------
# 6. Checkpointer (Persistence)
# -------------------

conn = sqlite3.connect("chatbot.db", check_same_thread=False)
checkpointer = SqliteSaver(conn=conn)


# -------------------
# 7. Graph
# -------------------

graph = StateGraph(ChatState)

graph.add_node("chat_node", chat_node)
graph.add_node("tools", tool_node)

graph.add_edge(START, "chat_node")
graph.add_conditional_edges("chat_node", tools_condition)
graph.add_edge("tools", "chat_node")

chatbot = graph.compile(checkpointer=checkpointer)


# -------------------
# 8. Helpers
# -------------------

def retrieve_all_threads():
    all_threads = set()
    for checkpoint in checkpointer.list(None):
        all_threads.add(checkpoint.config["configurable"]["thread_id"])
    return list(all_threads)


def thread_has_document(thread_id: str) -> bool:
    return str(thread_id) in _THREAD_RETRIEVERS


def thread_document_metadata(thread_id: str) -> dict:
    return _THREAD_METADATA.get(str(thread_id), {})