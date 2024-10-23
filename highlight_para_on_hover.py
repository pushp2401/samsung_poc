import streamlit as st

def highlight_paragraphs(text):
    # Split the text into paragraphs
    paragraphs = text.split('\n\n')
    
    # Create HTML content with red symbols at the end of each line
    html_content = ''
    for paragraph in paragraphs:
        lines = paragraph.split('\n')
        for i, line in enumerate(lines):
            html_content += f'<span class="line">{line} •</span><br>'
        html_content += '<br>'
    
    # Add custom CSS for styling and hover effect
    css = """
    <style>
    .container {
        font-family: Arial, sans-serif;
        max-width: 800px;
        margin: auto;
    }
    .line {
        display: block;
        margin-bottom: 5px;
        transition: background-color 0.3s ease;
    }
    .line:hover {
        background-color: rgba(255, 0, 0, 0.2);
    }
    </style>
    """
    
    # Combine HTML and CSS
    return f'<div class="container">{css}{html_content}</div>'

# Sample text
text = """Lorem ipsum dolor sit amet, consectetur adipiscing elit. Sed do eiusmod tempor incididunt ut labore et dolore magna aliqua. Ut enim ad minim veniam, quis nostrud exercitation ullamco laboris nisi ut aliquip ex ea commodo consequat.

Duis aute irure dolor in reprehenderit in voluptate velit esse cillum dolore eu fugiat nulla pariatur. Excepteur sint occaecat cupidatat non proident, sunt in culpa qui officia deserunt mollit anim id est laborum."""

# Create Streamlit app
st.title("Hover Effect App")

# Display the formatted text with hover effect
st.markdown(highlight_paragraphs(text), unsafe_allow_html=True)


# import streamlit as st
# import re
# from typing import List, Tuple

# def check_similarity(sentence1: str, sentence2: str) -> bool:
#     """Simple string comparison (you may want to use a more sophisticated NLP technique)"""
#     return sentence1.lower().find(sentence2.lower()) != -1 or sentence2.lower().find(sentence1.lower()) != -1

# def highlight_related_sentences(paragraph: str, answer: str) -> Tuple[str, List[Tuple[int, int]]]:
#     paragraph_sentences = re.split(r'[.!?]', paragraph.strip())
#     answer_sentences = re.split(r'[.!?]', answer.strip())

#     highlighted_paragraph = ""
#     related_ranges = []

#     for i, p_sentence in enumerate(paragraph_sentences):
#         if p_sentence:  # Skip empty strings
#             for j, a_sentence in enumerate(answer_sentences):
#                 if a_sentence and check_similarity(p_sentence, a_sentence):
#                     start_idx = paragraph.index(p_sentence)
#                     end_idx = start_idx + len(p_sentence)
#                     highlighted_paragraph += f"<mark>{p_sentence}</mark>"
#                     related_ranges.append((start_idx, end_idx))
#                     break
#             else:
#                 highlighted_paragraph += p_sentence

#     return highlighted_paragraph, related_ranges

# def main():
#     st.title("Related Sentence Highlighter")

#     col1, col2 = st.columns(2)

#     with col1:
#         paragraph = st.text_area("Enter Paragraph", height=300)
    
#     with col2:
#         answer = st.text_area("Enter Answer", height=300)

#     if st.button("Highlight"):
#         highlighted_paragraph, _ = highlight_related_sentences(paragraph, answer)
        
#         st.subheader("Highlighted Paragraph")
#         st.markdown(highlighted_paragraph, unsafe_allow_html=True)

#         st.subheader("Answer with Related Sentences Marked")
#         answer_sentences = re.split(r'[.!?]', answer.strip())
#         for i, sentence in enumerate(answer_sentences):
#             if sentence:  # Skip empty strings
#                 if any(check_similarity(sentence, p_sentence) for p_sentence in re.split(r'[.!?]', paragraph.strip())):
#                     st.markdown(f"* {sentence}. 🔴", unsafe_allow_html=True)
#                 else:
#                     st.markdown(f"* {sentence}.", unsafe_allow_html=True)

# if __name__ == "__main__":
#     main()
