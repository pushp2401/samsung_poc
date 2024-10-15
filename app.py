import os
import PyPDF2
import streamlit as st
from io import StringIO
from langchain.vectorstores import FAISS
from langchain.retrievers import SVMRetriever
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from dotenv import load_dotenv
from groq import Groq  # Import Groq

load_dotenv() 

# Initialize Groq client
client = Groq(api_key=os.environ.get("GROQ_API_KEY"))

# Streamlit page configuration
st.set_page_config(page_title="Document Processing with Groq", page_icon="📄")

@st.cache_data
def load_docs(files):
    st.info("Reading documents...")
    all_text = ""
    for file_path in files:
        file_extension = os.path.splitext(file_path.name)[1]
        if file_extension == ".pdf":
            pdf_reader = PyPDF2.PdfReader(file_path)
            text = ""
            for page in pdf_reader.pages:
                text += page.extract_text()
            all_text += text
        elif file_extension == ".txt":
            stringio = StringIO(file_path.getvalue().decode("utf-8"))
            text = stringio.read()
            all_text += text
        else:
            st.warning('Please provide .txt or .pdf files.', icon="⚠️")
    return all_text

@st.cache_resource
def create_retriever(_embeddings, splits, retriever_type):
    if retriever_type == "SIMILARITY SEARCH":
        vectorstore = FAISS.from_texts(splits, _embeddings)
        retriever = vectorstore.as_retriever(k=5)
    elif retriever_type == "SUPPORT VECTOR MACHINES":
        retriever = SVMRetriever.from_texts(splits, _embeddings)
    return retriever

@st.cache_resource
def split_texts(text, chunk_size, overlap):
    st.info("Splitting document...")
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=overlap)
    splits = text_splitter.split_text(text)
    if not splits:
        st.error("Failed to split document")
        st.stop()
    return splits

def get_groq_response(user_query, context, model):
    # Create the messages for Groq completion
    messages = [
        {
            "role": "user",
            "content": f"As a helpful assistant, answer the user's question if it's in the context. CONTEXT: {context} USER QUERY: {user_query}"
        }
    ]

    # Use the Groq client to get the completion with the selected model
    chat_completion = client.chat.completions.create(
        messages=messages,
        model=model,
    )
    
    return chat_completion.choices[0].message.content

# Main app logic
uploaded_files = st.file_uploader("Upload your documents", type=["pdf", "txt"], accept_multiple_files=True)
chunk_size = st.slider("Select chunk size", 100, 2000, 1000)
overlap = st.slider("Select overlap size", 0, 100, 0)
retriever_type = st.selectbox("Select retriever type", ["SIMILARITY SEARCH", "SUPPORT VECTOR MACHINES"])
embedding_option = st.selectbox("Select embedding option", ["OpenAI Embeddings"])

models = ["llama-3.1-70b-versatile"]
selected_model = st.selectbox("Select model for chat completion", models)

if uploaded_files:
    loaded_text = load_docs(uploaded_files)
    st.write("Documents uploaded and processed.")
    splits = split_texts(loaded_text, chunk_size, overlap)
    embeddings = OpenAIEmbeddings()
    retriever = create_retriever(embeddings, splits, retriever_type) 

    user_query = st.text_input("Ask a question about the uploaded documents:")
    # vector_store = FAISS.load_local("/Users/pushpanjali/samsung/congpt-async-claudetools/KCSFaissVectorStore", embeddings, allow_dangerous_deserialization=True)

    if user_query:
        # Retrieve relevant information
        result = retriever.get_relevant_documents(user_query)
        # result = vector_store.similarity_search(user_query)
        matched_info = ' '.join([doc.page_content for doc in result])
        context = f"Information: {matched_info}"
        
        # Generate response using Groq with the selected model
        response = get_groq_response(user_query, context, selected_model)
        
        st.write("Response:")
        st.write(response)  # Display response in Korean (Groq will handle it)