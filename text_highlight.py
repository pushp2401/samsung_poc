import streamlit as st
from difflib import SequenceMatcher
from sentence_transformers import SentenceTransformer, util 
import re 

def sen_sim_calc(s1 , s2) :
    model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

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

def highlight_similar_sentences(text1, text2):
    print("spliiting text")
    lines1 = split_into_sentences(text1)
    lines2 = split_into_sentences(text2)
    print("splitting done")
    html_output = ""
    
    for line1 in lines1:
        for line2 in lines2:
            # ratio = SequenceMatcher(None, line1, line2).ratio()
            print("calculating score")
            ratio = sen_sim_calc(line1 , line2)
            print("score calculation done") 
            print("-----------------------------")
            if ratio > 0.7:  # Adjust this threshold as needed
                html_output += f"<span style='background-color: yellow;'>{line1}</span><br>"
                break
        else:
            html_output += f"{line1}<br>"
    
    return html_output

st.title("Similar Sentence Highlighter")

col1, col2 = st.columns(2)

text1 = col1.text_area("Text Column 1", height=300)
text2 = col2.text_area("Text Column 2", height=300)

html1 = highlight_similar_sentences(text1, text2)
# html2 = highlight_similar_sentences(text2, text1)

col1.markdown(html1, unsafe_allow_html=True)
col2.markdown(html2, unsafe_allow_html=True)
