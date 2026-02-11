import streamlit as st
from streamlit_basic2_backend import workflow
from langchain_core.messages import HumanMessage

config={'configurable':{'thread_id':'thread_id_1'}}
#session_state store dictionary
if 'message_history' not in st.session_state:
    st.session_state['message_history']=[]

#message_history=[]  if we just use message_history and not not use session state whole code will  re-run whenever we press enter 
# that is why message_history alone not able to store   

for message in st.session_state['message_history']:
    with st.chat_message (message['role']):
        st.text (message ['content'])

user_input = st.chat_input ("type here")#input box with placeholder 
if user_input:
    st.session_state['message_history'].append ({'role':'user','content':user_input})
    with st.chat_message ('user'):#creates the box with emoji and emoji will be based on   ( whose message is this user or assistant )
        st.text (user_input)
    
        response = workflow.invoke(
        {"messages": [HumanMessage(content=user_input)]},
        config=config)
    ai_message=response["messages"][-1].content
    st.session_state['message_history'].append ({'role':'assistant','content':ai_message})
    with st.chat_message ('assistant'):#creates the box with emoji and emoji will be based on   ( whose message is this user or assistant )
        st.text (ai_message)# text in the above box

