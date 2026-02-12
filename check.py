import streamlit as st
import time

def generate_text():
    text = "Hello Basant, this is streaming demo."
    for word in text.split():
        time.sleep(0.3)
        yield word + " "

st.title("Streaming Demo")

st.write_stream(generate_text())
