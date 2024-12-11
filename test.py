import streamlit as st
import ollama
from PIL import Image

# Initialize session state
if 'ocr_result' not in st.session_state:
    st.session_state['ocr_result'] = None  # For storing OCR results

if 'chat_history' not in st.session_state:
    st.session_state['chat_history'] = []  # For storing chat history

if 'user_input' not in st.session_state:
    st.session_state['user_input'] = ""  # For storing current user input

# Page configuration
st.set_page_config(
    page_title="Troy's Sample Chat Bot",
    page_icon="🐒",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Title and description in main area
st.title("🐒 Troy's Sample Offline Chat Bot")

# Add clear button to top right
col1, col2 = st.columns([6, 1])
with col2:
    if st.button("Force Clear 🗑️"):
        st.session_state['ocr_result'] = None
        st.session_state['chat_history'] = []
        st.session_state['user_input'] = ""
        st.rerun()

st.markdown("---")

# Chat Input
st.subheader("Ask Anything!")
user_input = st.text_input(
    "Enter your message:",
    value=st.session_state['user_input'],  # Use the persisted value
    key="user_input",
    on_change=lambda: None  # Placeholder to avoid resetting on rerun
)

# Process user input when it's not empty
if user_input:
    # Append user message to the chat history
    st.session_state['chat_history'].append({"role": "user", "message": user_input})

    try:
        with st.spinner("Generating response..."):
            response = ollama.chat(
                model='llama3',
                messages=st.session_state['chat_history']
            )
            # Append assistant's response to the chat history
            st.session_state['chat_history'].append({"role": "assistant", "message": response.message.content})

            # Clear the input field after processing
            st.session_state['user_input'] = ""
            st.experimental_rerun()  # Force rerun to refresh UI
    except Exception as e:
        st.error(f"Error during chat: {str(e)}")

# Display Chat History
for chat in st.session_state['chat_history']:
    if chat['role'] == "user":
        st.markdown(f"**You:** {chat['message']}")
    else:
        st.markdown(f"**Bot:** {chat['message']}")

st.markdown("---")
