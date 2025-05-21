import streamlit as st
import pandas as pd
import os
from typing import List, Dict

# Import the RAGChatbot class from our main module
from rag_chatbot import RAGChatbot

# Set page configuration
st.set_page_config(
    page_title="RAG Chatbot",
    page_icon="🤖",
    layout="wide"
)

# Create Streamlit UI
st.title("📚 RAG Chatbot with LangChain")
st.markdown("""
This application demonstrates a Retrieval-Augmented Generation (RAG) chatbot 
that uses external data to answer questions accurately and contextually.
""")

# Sidebar for configuration
with st.sidebar:
    st.header("Configuration")
    
    # API Key input
    api_key = st.text_input("OpenAI API Key", type="password", help="Enter your OpenAI API key here")
    
    # Model selection
    model_name = st.selectbox(
        "Select LLM Model",
        options=["gpt-3.5-turbo", "gpt-4"],
        index=0,
        help="Select the OpenAI model to use"
    )
    
    # File uploader
    uploaded_file = st.file_uploader(
        "Upload knowledge base file", 
        type=["csv", "txt"],
        help="Upload a CSV or TXT file to use as knowledge base"
    )
    
    file_type = None
    if uploaded_file is not None:
        file_type = uploaded_file.name.split(".")[-1].lower()
        # Save the uploaded file temporarily
        with open("temp_data." + file_type, "wb") as f:
            f.write(uploaded_file.getbuffer())
    
    # Initialize button
    initialize_button = st.button("Initialize Chatbot", disabled=(not api_key or not uploaded_file))

# Main chat interface
st.header("Chat")

# Initialize session state
if "messages" not in st.session_state:
    st.session_state.messages = []
    
if "chatbot" not in st.session_state:
    st.session_state.chatbot = None
    
if "initialized" not in st.session_state:
    st.session_state.initialized = False

# Display messages from history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Initialize the chatbot when button is clicked
if initialize_button and not st.session_state.initialized:
    with st.spinner("Initializing chatbot..."):
        try:
            # Create RAG chatbot
            chatbot = RAGChatbot(api_key=api_key, model_name=model_name)
            
            # Load and process data
            documents = chatbot.load_and_process_data(f"temp_data.{file_type}", file_type)
            
            # Build vector store
            chatbot.build_vectorstore(documents)
            
            # Set up RAG pipeline
            chatbot.setup_rag_pipeline()
            
            # Store chatbot in session state
            st.session_state.chatbot = chatbot
            st.session_state.initialized = True
            
            # Add system message
            st.session_state.messages.append({
                "role": "assistant",
                "content": "Chatbot initialized successfully! I'm ready to answer your questions based on the uploaded knowledge base."
            })
            st.rerun()
        except Exception as e:
            st.error(f"Error initializing chatbot: {str(e)}")

# Handle user input
if st.session_state.initialized:
    if prompt := st.chat_input("Ask a question about your data"):
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": prompt})
        
        # Display user message
        with st.chat_message("user"):
            st.markdown(prompt)
            
        # Generate a response
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                try:
                    response = st.session_state.chatbot.ask(prompt)
                    st.markdown(response)
                    
                    # Add assistant response to chat history
                    st.session_state.messages.append({"role": "assistant", "content": response})
                except Exception as e:
                    error_msg = f"Error generating response: {str(e)}"
                    st.error(error_msg)
                    st.session_state.messages.append({"role": "assistant", "content": error_msg})

# Add option to save conversation
if st.sidebar.button("Save Conversation", disabled=not st.session_state.initialized or len(st.session_state.messages) == 0):
    # Convert messages to the format expected by save_conversation
    conversations = []
    
    for i in range(0, len(st.session_state.messages), 2):
        if i+1 < len(st.session_state.messages):
            conversations.append({
                "question": st.session_state.messages[i]["content"],
                "answer": st.session_state.messages[i+1]["content"]
            })
    
    # Save conversations
    try:
        st.session_state.chatbot.save_conversation(conversations, "saved_conversations.csv")
        st.sidebar.success("Conversation saved to saved_conversations.csv")
    except Exception as e:
        st.sidebar.error(f"Error saving conversation: {str(e)}")

# Display instructions if not initialized
if not st.session_state.initialized:
    st.info("To start using the chatbot, please upload a knowledge base file and provide your OpenAI API key in the sidebar, then click 'Initialize Chatbot'.")