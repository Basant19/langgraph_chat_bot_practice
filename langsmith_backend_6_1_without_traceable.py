from typing import TypedDict, Annotated
from langgraph.graph import StateGraph, START,END
from langgraph.graph.message import add_messages
from langgraph.checkpoint.sqlite import SqliteSaver
from langchain.chat_models import init_chat_model
from langchain_core.messages import HumanMessage
import sqlite3
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

#----------------establish connection----------------------------


# -------- Graph --------
graph = StateGraph(ChatState)
graph.add_node("chatbot", chatbot_node)
graph.add_edge(START, "chatbot")
graph.add_edge("chatbot",END)
# -------- Memory connection with Database --------
#note Sqlite works on single thread so that is why check_same_thread is kept false by default check_same_thread is true 
conn=sqlite3.connect (database='chatbot.db',check_same_thread=False)
checkpointer=SqliteSaver(conn=conn) #established connection

workflow = graph.compile(checkpointer=checkpointer)
def retrieve_all_threads():
    #extract every unique thread check point with None beacuse we want to extract every thread
    all_threads=set()#beause we want unique element
    for checkpoint in  checkpointer.list(None):
        all_threads.add (checkpoint.config["configurable"]['thread_id'])
    return list(all_threads)

'''#backend test 
CONFIG={'configurable':{'thread_id':'test_threaid_1'}}
response=workflow.invoke(
           {'messages':[HumanMessage('what is my name ')]},
            config=CONFIG
            
)
print (response)'''