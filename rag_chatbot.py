"""
RAG Chatbot using LangChain and OpenAI
Author: SID
Date: May 19, 2025

This module implements a Retrieval-Augmented Generation (RAG) chatbot
that uses LangChain components to provide accurate responses from a
knowledge base.
"""

import os
import pandas as pd
import numpy as np
from typing import List, Dict, Any

# LangChain imports
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_community.document_loaders import CSVLoader, TextLoader
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_text_splitters import RecursiveCharacterTextSplitter

class RAGChatbot:
    """
    A Retrieval-Augmented Generation (RAG) chatbot that uses LangChain components
    to retrieve relevant information from a knowledge base and generate responses.
    """
    
    def __init__(self, api_key: str, model_name: str = "gpt-3.5-turbo"):
        """
        Initialize the RAG chatbot.
        
        Args:
            api_key (str): OpenAI API key
            model_name (str): Name of the OpenAI model to use
        """
        os.environ["OPENAI_API_KEY"] = api_key
        self.embeddings = OpenAIEmbeddings()
        self.llm = ChatOpenAI(model_name=model_name, temperature=0)
        self.vectorstore = None
        self.retriever = None
        self.rag_chain = None
        
    def load_and_process_data(self, file_path: str, file_type: str = "csv"):
        """
        Load data from a file and process it for RAG.
        
        Args:
            file_path (str): Path to the data file
            file_type (str): Type of file ('csv' or 'txt')
            
        Returns:
            List[Document]: List of processed documents
        """
        print(f"Loading data from {file_path}...")
        
        # Load documents based on file type
        if file_type.lower() == "csv":
            loader = CSVLoader(file_path=file_path)
        elif file_type.lower() == "txt":
            loader = TextLoader(file_path=file_path)
        else:
            raise ValueError(f"Unsupported file type: {file_type}. Use 'csv' or 'txt'.")
            
        documents = loader.load()
        
        # Split documents into chunks
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200
        )
        split_documents = text_splitter.split_documents(documents)
        
        print(f"Processed {len(split_documents)} document chunks.")
        return split_documents
    
    def build_vectorstore(self, documents: List[Document]):
        """
        Build a vector store from processed documents.
        
        Args:
            documents (List[Document]): List of processed documents
        """
        print("Building vector store...")
        self.vectorstore = FAISS.from_documents(documents, self.embeddings)
        self.retriever = self.vectorstore.as_retriever(
            search_type="similarity",
            search_kwargs={"k": 3}
        )
        print("Vector store built successfully.")
    
    def setup_rag_pipeline(self):
        """
        Set up the RAG pipeline with LangChain components.
        """
        # Define the RAG prompt template
        template = """
        You are a helpful AI assistant. Use the following retrieved context to answer the user's question.
        If you don't know the answer based on the context, just say you don't know. Don't make up information.
        
        Context:
        {context}
        
        Question: {question}
        
        Answer:
        """
        
        prompt = ChatPromptTemplate.from_template(template)
        
        # Set up the RAG chain
        self.rag_chain = (
            {"context": self.retriever, "question": RunnablePassthrough()}
            | prompt
            | self.llm
            | StrOutputParser()
        )
        
        print("RAG pipeline setup complete.")
    
    def format_docs(self, docs: List[Document]) -> str:
        """
        Format a list of documents into a single string.
        
        Args:
            docs (List[Document]): List of retrieved documents
            
        Returns:
            str: Formatted string of document contents
        """
        return "\n\n".join([doc.page_content for doc in docs])
    
    def ask(self, question: str) -> str:
        """
        Ask a question to the RAG chatbot.
        
        Args:
            question (str): User's question
            
        Returns:
            str: Generated response
        """
        if not self.rag_chain:
            raise ValueError("RAG pipeline is not set up. Call setup_rag_pipeline() first.")
            
        return self.rag_chain.invoke(question)

    def save_conversation(self, conversations: List[Dict[str, str]], file_path: str):
        """
        Save conversations to a file.
        
        Args:
            conversations (List[Dict[str, str]]): List of conversation dictionaries
            file_path (str): Path to save the file
        """
        df = pd.DataFrame(conversations)
        df.to_csv(file_path, index=False)
        print(f"Conversations saved to {file_path}")
        
# Demo usage
if __name__ == "__main__":
    # Initialize chatbot
    api_key = "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"  # Replace with your actual API key
    chatbot = RAGChatbot(api_key=api_key)
    
    # Load and process data
    documents = chatbot.load_and_process_data("sample_data.csv", "csv")
    
    # Build vector store
    chatbot.build_vectorstore(documents)
    
    # Set up RAG pipeline
    chatbot.setup_rag_pipeline()
    
    # Ask questions
    questions = [
        "What are the main benefits of RAG?",
        "How does LangChain help with implementing RAG?",
        "What are the limitations of traditional LLMs without RAG?"
    ]
    
    # Store conversations
    conversations = []
    
    for question in questions:
        answer = chatbot.ask(question)
        print(f"Q: {question}")
        print(f"A: {answer}\n")
        
        conversations.append({
            "question": question,
            "answer": answer
        })
    
    # Save conversations
    chatbot.save_conversation(conversations, "sample_conversations.csv")
