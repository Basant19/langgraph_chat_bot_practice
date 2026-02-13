#example to streaming 

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
    
    with st.chat_message ('assistant'):
        ai_message= st.write_stream(
        message_chunk.content for message_chunk,meta_data in  workflow.stream(
            {'messages':[HumanMessage(user_input)]},
            config={'configurable':{'thread_id':'thread_id_1'}},
            stream_mode='messages'
        ))
    
    st.session_state['message_history'].append ({'role':'assistant','content':ai_message})