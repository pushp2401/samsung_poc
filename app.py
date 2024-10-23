import os
import PyPDF2
import streamlit as st
from io import StringIO
# from langchain.vectorstores import FAISS
from langchain_community.vectorstores import FAISS
from langchain.retrievers import SVMRetriever
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
# from langchain.embeddings import HuggingFaceEmbeddings
from dotenv import load_dotenv 
from groq import Groq  # Import Groq
import streamlit as st
# from st_files_connection import FilesConnection
import boto3
import tempfile
import shutil
import re
from sentence_transformers import SentenceTransformer, util 
import time

load_dotenv()  


# Initialize Groq client
client = Groq(api_key=os.environ.get("GROQ_API_KEY"))
# Streamlit page configuration
st.set_page_config(page_title="KCS database query with groq", page_icon="📄")
st.title(":red[KCS] knowledge base query tool")

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
            # "content": f"As a helpful assistant, answer the user's question if it's in the context. CONTEXT: {context} USER QUERY: {user_query}" 
            "content": f"You are an intelligent and helpful construction assistant,providing information on civil engineering, construction practices, safety regulations etc. Answer the user's question based on the given context. CONTEXT: {context} USER QUERY: {user_query}" 

        }
    ]

    # Use the Groq client to get the completion with the selected model
    chat_completion = client.chat.completions.create(
        messages=messages,
        model=model,
    )
    
    return chat_completion.choices[0].message.content

    
from langchain_community.vectorstores import FAISS
import tempfile
from botocore.exceptions import ClientError, NoCredentialsError
import os
import boto3
def load_vectorstore():
    s3 = boto3.client('s3')
    bucket_name = os.getenv("AWS_BUCKET_NAME")
    folder_prefix = 'vector-store/2024-10-16'
    
    try:
        with tempfile.TemporaryDirectory() as temp_dir:
            # List objects in the S3 bucket
            paginator = s3.get_paginator('list_objects_v2')
            page_iterator = paginator.paginate(Bucket=bucket_name, Prefix=folder_prefix)
            
            # Download filtered files
            for page in page_iterator:
                for obj in page.get('Contents', []):
                    key = obj['Key']
                    
                    # Apply file filter if specified
                    if os.path.basename(key) not in ['index.faiss', 'index.pkl']:
                        continue
                    
                    # Construct local file path
                    local_path = os.path.join(temp_dir , os.path.basename(key))
                    os.makedirs(os.path.dirname(local_path), exist_ok=True)
                    try:
                        print(f"Downloading: {key} -> {local_path}")
                        s3.download_file(bucket_name, key, local_path)
                        print(f"Downloaded: {local_path}")
                    except Exception as e:
                        print(f"Error downloading {key}: {e}")
            embeddings = OpenAIEmbeddings(model="text-embedding-3-large") 
            
            vectorstore = FAISS.load_local(temp_dir, embeddings=embeddings, allow_dangerous_deserialization=True)
        return vectorstore
        
    except NoCredentialsError:
        print("No AWS credentials found.")
    except Exception as e:
        print(f"An error occurred: {e}")
# Main app logic
# uploaded_files = st.file_uploader("Upload your documents", type=["pdf", "txt"], accept_multiple_files=True)
# chunk_size = st.slider("Select chunk size", 100, 2000, 1000)
# overlap = st.slider("Select overlap size", 0, 100, 0)
# retriever_type = st.selectbox("Select retriever type", ["SIMILARITY SEARCH", "SUPPORT VECTOR MACHINES"])
# embedding_option = st.selectbox("Select embedding option", ["OpenAI Embeddings"])

models = ["llama-3.1-70b-versatile" , "llama-3.1-8b-instant" , "gemma2-9b-it"]
selected_model = st.selectbox("Select model for chat completion", models)


# loaded_text = load_docs(uploaded_files)
# st.write("Documents uploaded and processed.")
# splits = split_texts(loaded_text, chunk_size, overlap)
# retriever = create_retriever(embeddings, splits, retriever_type) 
# vector_store = download_vector_store('congpt' , 'vector-store/2024-10-16') 
# vector_store = load_vectorstore()
if "vector_store" not in st.session_state :
    # vector_store = FAISS.load_local("/Users/pushpanjali/samsung/congpt-async-claudetools/KCSFaissVectorStore" , embeddings ,  allow_dangerous_deserialization=True)
    with st.spinner("loading vector store") : 
        vector_store = load_vectorstore()
# retriever = vector_store.as_retriever()
# user_query = st.chat_input("Ask a question about the uploaded documents:") 
# vector_store = FAISS.load_local("/Users/pushpanjali/samsung/congpt-async-claudetools/KCSFaissVectorStore", embeddings, allow_dangerous_deserialization=True)
# import boto3

# s3 = boto3.client('s3')
# bucket_name = 'congpt'
# vector_store_path = 'vector-store/2024-10-16'
# vector_store_path = download_folder('congpt' , 'vector-store/2024-10-16' ,'faiss_tmp' )
# vector_store = FAISS.load_local(vector_store_path , embeddings, allow_dangerous_deserialization=True)
model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

def sen_sim_calc(s1 , s2) :

    #Compute embedding for both lists
    embedding_1= model.encode(s1 , convert_to_tensor=False)
    embedding_2 = model.encode(s2 , convert_to_tensor=False)

    score = util.pytorch_cos_sim(embedding_1, embedding_2) 
    return score

def split_into_sentences(paragraph):
    # Regular expression pattern
    sentence_endings = r'(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<=\.|\?|\!)\s'
    sentences = re.split(sentence_endings, paragraph)    
    return sentences
def update_string(new_value):
    st.session_state.response = new_value
    st.write(st.session_state.response)

def highlight_similar_sentences(resp, chunk , highlight_color):
    # print("spliiting text")
    resp_lines = split_into_sentences(resp)
    chunk_lines = split_into_sentences(chunk)
    # print("splitting done")
    resp_output = ""
    chunk_output = ""
    
    for line1 in resp_lines:
        for line2 in chunk_lines:
            # ratio = SequenceMatcher(None, line1, line2).ratio()
            # print("calculating score")
            ratio = sen_sim_calc(line1 , line2)
            # print("score calculation done") 
            # print("-----------------------------")
            if ratio >= 0.85:  # Adjust this threshold as needed
                # code for highlight paragraphs at the same time 
                resp_output += f"<span style='background-color:{highlight_color};'>{line1}</span><br>"
                chunk_output += f"<span style='background-color: {highlight_color};'>{line2}</span><br>"
                #-------------------------------------------------
                # resp_output += f"{line1} &#9731" 
                # chunk_output += f"<span style='background-color: {highlight_color};'>{line2}</span><br>"
        else:
            resp_output += f"{line1}<br>"
            chunk_output += f"{line2}<br>"
    
    return resp_output , chunk_output


# def highlight_paragraphs(text):
#     # Split the text into paragraphs
#     paragraphs = text.split('\n\n')
    
#     # Create HTML content with red symbols at the end of each line
#     html_content = ''
#     for paragraph in paragraphs:
#         lines = paragraph.split('\n')
#         for i, line in enumerate(lines):
#             html_content += f'<span class="line">{line} •</span><br>'
#         html_content += '<br>'
    
#     # Add custom CSS for styling and hover effect
#     css = """
#     <style>
#     .container {
#         font-family: Arial, sans-serif;
#         max-width: 800px;
#         margin: auto;
#     }
#     .line {
#         display: block;
#         margin-bottom: 5px;
#         transition: background-color 0.3s ease;
#     }
#     .line:hover {
#         background-color: rgba(255, 0, 0, 0.2);
#     }
#     </style>
#     """
    
#     # Combine HTML and CSS
#     return f'<div class="container">{css}{html_content}</div>'


if user_query := st.chat_input("Ask a question about KCS documents:") : 
    # import pdb; pdb.set_trace()
    if user_query :
        st.write(f"QUESTION : {user_query}")
    # Retrieve relevant information
    result = vector_store.similarity_search(user_query)
    matched_info = ' '.join([doc.page_content for doc in result])
    context = f"Information: {matched_info}"
    # if 'response' not in st.session_state : 
        # st.session_state.response = ""
    # st.session_state.response = get_groq_response(user_query, context, selected_model)

    response = get_groq_response(user_query, context, selected_model)
    placeholder = st.empty()
    placeholder.write(response)
    # with placeholder.container() : 
    # st.write("Response:")
    # st.write(response)  # Display response in Korean (Groq will handle it)

    #### working code part ##############################
    # resp_lines = split_into_sentences(response)
    # cl1 , cl2 , cl3 , cl4 = split_into_sentences(result[0].page_content),split_into_sentences(result[1].page_content),split_into_sentences(result[2].page_content),split_into_sentences(result[3].page_content)
    updated_resp = ""
    with st.sidebar : 
        with st.container() :
            st.header("Generated answer references" , divider = "red")
            with st.expander(str(result[0].metadata["filename"])) : 
                r1 ,c1 = highlight_similar_sentences(response , result[0].page_content , 'LightBlue') 
                # st.write(str(result[0].page_content)) 
                st.write(c1 , unsafe_allow_html = True)
                updated_resp = updated_resp + r1
            with st.expander(str(result[1].metadata["filename"])) : 
                r2 ,c2 = highlight_similar_sentences(response , result[1].page_content , 'LightBlue') 
                # st.write(str(result[1].page_content)) 
                st.write(c2, unsafe_allow_html = True)
                updated_resp = updated_resp + '\n' + r2 
            with st.expander(str(result[2].metadata["filename"])) : 
                r3 ,c3 = highlight_similar_sentences(response , result[2].page_content , 'LightBlue') 
                # st.write(str(result[2].page_content)) 
                st.write(c3, unsafe_allow_html = True)
                updated_resp = updated_resp + '\n' + r3
            with st.expander(str(result[3].metadata["filename"])) : 
                r4 ,c4 = highlight_similar_sentences(response , result[3].page_content , 'LightBlue') 
                # st.write(str(result[3].page_content)) 
                st.write(c4, unsafe_allow_html = True)
                updated_resp = updated_resp + '\n' + r4 
    ###################################################################################  
    # time.sleep(1) 
    # placeholder.text_are("Response : " , "Here's a bouquet &mdash;\
                        #   :tulip::cherry_blossom::rose::hibiscus::sunflower::blossom:")
    # placeholder.write(updated_resp , unsafe_allow_html = True)
    