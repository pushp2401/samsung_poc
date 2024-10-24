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
from langchain_community.vectorstores import FAISS
import tempfile
from botocore.exceptions import ClientError, NoCredentialsError
import os
import boto3


load_dotenv()  

# count = 0


# Initialize Groq client
client = Groq(api_key=os.environ.get("GROQ_API_KEY"))
# Streamlit page configuration
st.set_page_config(page_title="KCS database query with groq", page_icon="📄")
st.title(":red[KCS] knowledge base query tool")

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

models = ["llama-3.1-70b-versatile" , "llama-3.1-8b-instant" , "gemma2-9b-it"]
selected_model = st.selectbox("Select model for chat completion", models)


# if "vector_store" not in st.session_state :
#     # vector_store = FAISS.load_local("/Users/pushpanjali/samsung/congpt-async-claudetools/KCSFaissVectorStore" , embeddings ,  allow_dangerous_deserialization=True)
#     with st.spinner("loading vector store") : 
#         vector_store = load_vectorstore()

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

# def highlight_similar_sentences(resp, chunk , highlight_color , count):
#     # import pdb; pdb.set_trace()
#     # print("spliiting text")
#     resp_lines = split_into_sentences(resp)
#     chunk_lines = split_into_sentences(chunk)
#     # print("splitting done")
#     resp_output = ""
#     chunk_output = ""
#     for line1 in resp_lines:
#         flag = True
#         for line2 in chunk_lines:
#             # ratio = SequenceMatcher(None, line1, line2).ratio()
#             # print("calculating score")
#             ratio = sen_sim_calc(line1 , line2)
#             # print("score calculation done") 
#             # print("-----------------------------")
#             ref_index = f"[{count}]"
#             if ratio >= 0.85:  # Adjust this threshold as needed
#                 # code for highlight paragraphs at the same time 
#                 # resp_output += f"<span style='background-color:{highlight_color};'>{line1} {ref_index}</span><br>"
#                 # import pdb; pdb.set_trace()
#                 resp_output+= f"{line1} <span style='background-color:{highlight_color};color:red'>{ref_index}</span><br>"
#                 chunk_output += f"<span style='background-color: {highlight_color};'>{line2}</span><br> <span style='background-color: red;'>{ref_index}</span><br>"
#                 count += 1
#             else : 
#                 chunk_output += f"{line2}<br>" 
#             if ratio < 0.85  :
#                 if flag :
#                     resp_output += f"{line1}<br>"
#                     flag = False



#         # else:
#         #     chunk_output += f"{line2}<br>" 
#         #     resp_output += f"{line1}<br>"

#     return resp_output , chunk_output , count

def highlight_similar_sentences(resp , chunk , highlight_color , count) :
    # import pdb; pdb.set_trace()
    resp_lines = split_into_sentences(resp)
    chunk_lines = split_into_sentences(chunk)
    resp_output = ""
    chunk_output = ""
    # import pdb; pdb.set_trace()
    for line1 in resp_lines:
        for line2 in chunk_lines :
            sim = sen_sim_calc(line1 , line2)
            ref_index = f"[{count}]" 
            if sim >= 0.87 : 
                # import pdb; pdb.set_trace()
                if line1 not in resp_output : 
                    resp_output+= f"{line1} <span style='background-color:{highlight_color};color:red'>{ref_index}</span><br>"
                if line2 not in chunk_output : 
                    # chunk_output+= f"{line2} <span style='background-color:{highlight_color};color:red'>{ref_index}</span><br>"
                    #   chunk_output = chunk_output[  : chunk_line2_start_index  ] + f"<span style='background-color: {highlight_color};'>" + line2 + f"</span><br> <span style='background-color: red;'>{ref_index}</span><br>" + chunk_output[chunk_line2_end_index : ]
                    chunk_output += f"<span style='background-color: {highlight_color};'>{line2}</span><br> <span style='background-color: red;'>{ref_index}</span><br>"
                
                if line1 in resp_output : 
                    end_index_line1 = resp_output.index(line1) + len(line1) - 1
                    resp_output = resp_output[ : end_index_line1 + 1] + f"<span style='background-color:{highlight_color};color:red'>{ref_index}</span>" + resp_output [end_index_line1+1 : ]
                if line2 in chunk_output  : 
                    # if f"<span style='background-color: {highlight_color};'>{line2}</span><br> <span style='background-color: red;'>{ref_index}</span><br>" in chunk_output : 
                    if (f"<span style='background-color: {highlight_color};'>{line2}</span><br>" in chunk_output) and (f"<span style='background-color: red;'>{ref_index}</span><br>" in chunk_output) :
                        pass
                    elif (f"<span style='background-color: {highlight_color};'>{line2}</span><br>" in chunk_output) and (f"<span style='background-color: red;'>{ref_index}</span><br>" not in chunk_output) :
                        end_index_line2 = chunk_output.index(line2) + len(line2) - 1 
                        chunk_output = chunk_output[ : end_index_line2 + 1] + f"<span style='background-color: red;'>{ref_index}</span>" + chunk_output [end_index_line2 + 1 : ]
                    else :
                        start_index_line2 = chunk_output.index(line2) 
                        end_index_line2 = chunk_output.index(line2) + len(line2) - 1 
                        chunk_output = chunk_output[ : start_index_line2] + f" <span style='background-color: {highlight_color};'>" + chunk_output[start_index_line2 : end_index_line2 + 1] + f"</span><br> <span style='background-color: red;'>{ref_index}</span><br>"
                count+=1
            else : 
                if line1 in resp_output : 
                    pass
                else : 
                    resp_output += f"{line1}<br>"
                if line2 in chunk_output : 
                    pass
                else : 
                    chunk_output += f"{line2}<br>" 
    return resp_output , chunk_output , count


# def highlight_similar_sentences(resp, chunk, highlight_color, count):

#     resp_lines = split_into_sentences(resp)
#     chunk_lines = split_into_sentences(chunk)
#     resp_output = []
#     chunk_output = []

#     for line1 in resp_lines:
#         for line2 in chunk_lines:
#             sim = sen_sim_calc(line1, line2)
#             ref_index = f"[{count}]"

#             if sim >= 0.87:
#                 count += 1
#                 resp_output.append(f"{line1} <span style='background-color:{highlight_color};color:red'>{ref_index}</span><br>")
#                 chunk_output.append(f"<span style='background-color: {highlight_color};'>{line2}</span><br> <span style='background-color: red;'>{ref_index}</span><br>")
#             else:
#                 resp_output.append(f"{line1}<br>")
#                 chunk_output.append(f"{line2}<br>")

#     return "".join(resp_output), "".join(chunk_output), count


if user_query := st.chat_input("Ask a question about KCS documents:") : 
    # import pdb; pdb.set_trace()
    if user_query :
        with st.chat_message("user") : 
            st.write(f"QUESTION : {user_query}")
    # Retrieve relevant information
    if "vector_store" not in st.session_state :
    # vector_store = FAISS.load_local("/Users/pushpanjali/samsung/congpt-async-claudetools/KCSFaissVectorStore" , embeddings ,  allow_dangerous_deserialization=True)
        with st.spinner("loading vector store") : 
            st.session_state.vector_store = load_vectorstore()
    with st.spinner("retrieving context") :
        result = st.session_state.vector_store.similarity_search(user_query)

    matched_info = ' '.join([doc.page_content for doc in result]) 
    context = f"Information: {matched_info}"
    # if 'response' not in st.session_state : 
        # st.session_state.response = ""
    # st.session_state.response = get_groq_response(user_query, context, selected_model)

    response = get_groq_response(user_query, context, selected_model)
    # with st.chat_message("ai") : 
    placeholder = st.empty()
    
    placeholder.write(response)

    # #### working code part ##############################
    updated_resp = ""
    with st.sidebar : 
        with st.container() :
            with st.spinner("Generating references") : 
                st.header("Generated answer references" , divider = "red")
                # full_context = " ####$#### ".join([doc.page_content for doc in result])
                # full_context = "----chunk1------" + "\n" + result[0].page_content + "\n" + "----chunk2------" + "\n" + result[1].page_content + "\n" +  "----chunk3------" + "\n" + result[2].page_content + "\n" + "----chunk4------" + "\n" + result[3].page_content
                # r , c , ref_count = highlight_similar_sentences(response , full_context , 'LightBlue' , 0)
                # st.write(c , unsafe_allow_html = True) 

    # placeholder.write(r , unsafe_allow_html = True) 
                # exit()
                
                with st.expander(str(result[0].metadata["filename"])) : 
                    # import pdb; pdb.set_trace() 
                    r1 ,c1 , ref_count1 = highlight_similar_sentences(response , result[0].page_content , 'LightBlue' , 0) 
                    # st.write(str(result[0].page_content)) 
                    st.write(c1 , unsafe_allow_html = True) 
                    updated_resp = updated_resp + r1
                with st.expander(str(result[1].metadata["filename"])) : 
                    # import pdb; pdb.set_trace() 
                    r2 ,c2 , ref_count2 = highlight_similar_sentences(r1 , result[1].page_content , 'LightBlue' , ref_count1) 
                    # st.write(str(result[1].page_content)) 
                    st.write(c2, unsafe_allow_html = True)
                    updated_resp = updated_resp + '\n' + r2 
                with st.expander(str(result[2].metadata["filename"])) : 
                    # import pdb; pdb.set_trace() 
                    r3 ,c3 , ref_count3 = highlight_similar_sentences(r2 , result[2].page_content , 'LightBlue' , ref_count2) 
                    # st.write(str(result[2].page_content)) 
                    st.write(c3, unsafe_allow_html = True)
                    updated_resp = updated_resp + '\n' + r3
                with st.expander(str(result[3].metadata["filename"])) : 
                    # import pdb; pdb.set_trace() 
                    r4 ,c4  , ref_count4 = highlight_similar_sentences(r3 , result[3].page_content , 'LightBlue' , ref_count3) 
                    # st.write(str(result[3].page_content)) 
                    st.write(c4, unsafe_allow_html = True)
                    # updated_resp = updated_resp + '\n' + r4 
    ###################################################################################  
    # time.sleep(1) 
    # placeholder.text_are("Response : " , "Here's a bouquet &mdash;\
                        #   :tulip::cherry_blossom::rose::hibiscus::sunflower::blossom:")
    # placeholder.write(updated_resp , unsafe_allow_html = True)
    placeholder.write(r4 , unsafe_allow_html = True) 


    
    
