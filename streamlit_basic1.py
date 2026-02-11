# building chat message of streamlit and chat input of streamlit 
import streamlit as st 

with st.chat_message('user'):
    st.text ('Hi')

with st.chat_message('assistant'):#creates the box with emoji and emoji will be based on   (whose message is this user or assistant)
    st.text ('How are you!')

user_input=st.chat_input('Type here ')#input box which placeholder 
if user_input:
    with st.chat_message ('user'):#input box  which placeholder 
        st.text (user_input)
