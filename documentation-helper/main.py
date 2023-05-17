from be.core import run_llm
import streamlit as st
from streamlit_chat import message

st.header("Langchain helper chat")
propmt = st.text_input('Chat whit me', placeholder='Enter a question ....')

if 'user_props_history' not in st.session_state:
    st.session_state['user_props_history'] = []

if 'chat_answers_history' not in st.session_state:
    st.session_state['chat_answers_history'] = []

if (propmt):
    with st.spinner("generating response ..."):
        generated_response = run_llm(query=propmt)

        formatted_response = (f"{generated_response['result']} \n\n ")

        st.session_state['user_props_history'].append(propmt)
        st.session_state['chat_answers_history'].append(formatted_response)

if st.session_state['chat_answers_history']:
    for  user_query,generated_response in zip(st.session_state['user_props_history'], st.session_state['chat_answers_history']):
        message(user_query, is_user=True)
        message(generated_response)