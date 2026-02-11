from typing import TypedDict, Annotated
from langgraph.graph import StateGraph, START,END
from langgraph.graph.message import add_messages
from langgraph.checkpoint.memory import InMemorySaver
from langchain.chat_models import init_chat_model
from langchain_core.messages import HumanMessage

from dotenv import load_dotenv
import os 
load_dotenv()

api_key=os.getenv("GOOGLE_API_KEY")
# -------- LLM --------
llm = init_chat_model("google_genai:gemini-2.5-flash",api_key=api_key)

# -------- State --------
class ChatState(TypedDict):
    messages: Annotated[list, add_messages]

# -------- Node --------
def chatbot_node(state: ChatState):
    response = llm.invoke(state["messages"])
    return {"messages": [response]}

# -------- Graph --------
graph = StateGraph(ChatState)
graph.add_node("chatbot", chatbot_node)
graph.add_edge(START, "chatbot")
graph.add_edge("chatbot",END)
# -------- Memory --------
memory = InMemorySaver()
workflow = graph.compile(checkpointer=memory)
