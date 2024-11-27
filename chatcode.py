import os
import json
import datetime
import time
from llama_index.llms.ollama import Ollama
import streamlit as st
import ollama
import re
import os
from llama_index.core import PromptTemplate
try:
    OLLAMA_MODELS = ollama.list()['models']
except Exception as e:
    st.warning("Please make sure Ollama is installed first.")
    st.stop()

OUTPUT_DIR = 'llm_conversations'
OUTPUT_DIR = os.path.join(os.getcwd(), OUTPUT_DIR)

def streamlit_ollama(model_name, user_question, chat_history_key):
    if chat_history_key not in st.session_state.keys():
        st.session_state[chat_history_key] = []
    print_chat_history_timeline(chat_history_key)

    if user_question:
        st.session_state[chat_history_key].append({"content": f"{user_question}", "role": "user"})
        with st.chat_message("question"):
            st.write(user_question)
        message = [dict(content=message["content"], role=message["role"]) for message in st.session_state[chat_history_key]]
        def llm_stream(response):
            response = ollama.chat(model_name, message, stream=True)
            for chunk in response:
                yield chunk['message']['content']
        with st.chat_message("response"):
            chat_box = st.empty()
            response_message = chat_box.write_stream(llm_stream(message))
        st.session_state[chat_history_key].append({"content": f"{response_message}", "role": "assistant"})
        return response_message

def print_chat_history_timeline(chat_history_key):
    for message in st.session_state[chat_history_key]:
        role = message['role']
        if role == 'user':
            with st.chat_message("user"):
                question = message['content']
                st.markdown(f"{question}", unsafe_allow_html=True)
        elif role == 'assistant':
            with st.chat_message("assistant"):
                st.markdown(message['content'], unsafe_allow_html=True)

def assert_models_installed():
    if len(OLLAMA_MODELS) < 1:
        st.sidebar.warning("No models found.")
        st.stop()

def select_model():
    model_names = [model['name'] for model in OLLAMA_MODELS]
    llm_name = st.sidebar.selectbox(f'Choose Model (available {len(model_names)})', [""] + model_names)
    if llm_name:
        llm_details = [model for model in OLLAMA_MODELS if model['name'] == llm_name][0]
        if type(llm_details['size']) != str:
            llm_details['size'] = f"{round(llm_details['size']/1e9,2)} GB"
        with st.expander('LLM Details'):
            st.write(llm_details)
    return llm_name

def generate_filename(content):
    model = Ollama(model='mistral-small:22b')
    prompt_template = PromptTemplate(
    """
    Summarize the context and return a title (no more than four words), no need markdown
    Context : {content}
    """
    )
    prompt = prompt_template.format(content=content)

    title = model.complete(prompt)
    return title.text.strip()

def save_conversation(llm_name, conversation_key):
    save = st.sidebar.button('Save conversation')
    if save:
        timestamp = datetime.datetime.now().strftime('%Y%m%d%H%M%S')
        if st.session_state[conversation_key]:
            content_summary = st.session_state[conversation_key][0]['content']
            descriptive_name = generate_filename(content_summary)
            filename = f"{descriptive_name}"
            
            if not os.path.exists(OUTPUT_DIR):
                os.makedirs(OUTPUT_DIR)
            
            with open(os.path.join(OUTPUT_DIR, f'{filename}.json'), "w") as f:
                json.dump(st.session_state[conversation_key], f, indent=4)
            st.success(f"Saved to {filename}.json")
            time.sleep(1.0)
            st.rerun()

def load_conversation_history():
    history_files = [f for f in os.listdir(OUTPUT_DIR) if f.endswith('.json')]
    history_files.sort(reverse=True)
    return history_files

def load_selected_conversation(filename):
    with open(os.path.join(OUTPUT_DIR, filename), 'r') as f:
        return json.load(f)

if __name__ == "__main__":
    st.set_page_config(layout='wide', page_title='Code with CodeLlama')
    st.sidebar.title('Chat')
    llm_name = select_model()
    assert_models_installed()
    if not llm_name: st.stop()
    conversation_key = f'model_{llm_name}'
    
    st.sidebar.title('History')
    history_files = load_conversation_history()
    selected_history = st.sidebar.selectbox("Select a previous conversation", ["Current Conversation"] + history_files)
    
    if selected_history != "Current Conversation":
        if st.sidebar.button("Load Selected Conversation"):
            st.session_state[conversation_key] = load_selected_conversation(selected_history)
            st.rerun()

    prompt = st.chat_input(f"Ask me a question ...")
    
    streamlit_ollama(llm_name, prompt, conversation_key)
    if st.session_state[conversation_key]:
        new_conversation = st.sidebar.button('New conversation')
        if new_conversation:
            st.session_state[conversation_key] = []
            st.rerun()
        save_conversation(llm_name, conversation_key)
