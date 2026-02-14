#example to streaming 

import streamlit as st
from streamlit_basic2_backend import workflow
from langchain_core.messages import HumanMessage
import uuid

#---------------utility function-----------------------------------------------------------------------------------------------------------
def generate_thread_id():
    thread_id=uuid.uuid4()
    return thread_id


#utiltiy function for new chat button which on click generate a new thread id and remove message from the window 
def reset_chat ():
    thread_id=generate_thread_id()
    st.session_state['thread_id']=thread_id
    st.session_state['message_history']=[] #to clean the message history to clean the window 


#----------adding session ---------------------------------------------------------------------------------------------------------------

#session_state store dictionary
if 'message_history' not in st.session_state:
    st.session_state['message_history']=[]

if 'thread_id' not in st.session_state:
    st.session_state['thread_id']=generate_thread_id()




#------------adding side bar------------------------------------------------------------------------------------------------------------
st.sidebar.title('Langgraph chatbot')

if st.sidebar.button('New Chat'):
    reset_chat()

st.sidebar.header ('My conversation')
st.sidebar.text (st.session_state['thread_id'])

#------------------------------adding older message in window --------------------------------------------------------------------------------------
#message_history=[]  if we just use message_history and not not use session state whole code will  re-run whenever we press enter 
# that is why message_history alone not able to store   
for message in st.session_state['message_history']:
    with st.chat_message (message['role']):
        st.text (message ['content'])
#-------------------------------adding newer message --------------------------------------------------------------------------------------------------------------
user_input = st.chat_input ("type here")#input box with placeholder 
if user_input:
    st.session_state['message_history'].append ({'role':'user','content':user_input})
    with st.chat_message ('user'):#creates the box with emoji and emoji will be based on   ( whose message is this user or assistant )
        st.text (user_input)

    CONFIG={'configurable':{'thread_id':st.session_state['thread_id']}}

    with st.chat_message ('assistant'):
        ai_message= st.write_stream(
        message_chunk.content for message_chunk,meta_data in  workflow.stream(
            {'messages':[HumanMessage(user_input)]},
            config=CONFIG,
            stream_mode='messages'
        ))
    
    st.session_state['message_history'].append ({'role':'assistant','content':ai_message})