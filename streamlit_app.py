import streamlit as st
from pypdf import PdfReader
from ingestion.chunker import chunk_pages
from ingestion.embeddings_store import create_vector_store
from ingestion.qa import answer_question

st.set_page_config(page_title="IMF Qatar RAG", layout="wide")

st.title("📊 IMF Qatar Article IV – RAG Demo")
st.write("Ask questions grounded in the IMF Qatar 2024 Article IV report.")

@st.cache_resource
def load_vector_store():
    reader = PdfReader("data/imf_qatar_report.pdf")
    pages = []
    for page_number, page in enumerate(reader.pages, start=1):
        text = page.extract_text()
        if text:
            pages.append({"page": page_number, "text": text})

    chunks = chunk_pages(pages)
    vector_store = create_vector_store(chunks)
    return vector_store

with st.spinner("Loading document and building index..."):
    vector_store = load_vector_store()

st.success("Vector store ready!")

question = st.text_input(
    "Enter your question:",
    placeholder="What are the key economic risks facing Qatar according to the IMF report?"
)

if st.button("Get Answer"):
    if question.strip():
        with st.spinner("Retrieving answer..."):
            answer = answer_question(vector_store, question)
        st.subheader("Answer")
        st.write(answer)
    else:
        st.warning("Please enter a question.")
