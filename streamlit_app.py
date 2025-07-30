import streamlit as st
from app.extract_text import extract_text_from_pdf
from app.embedding_store import get_chunks, get_vector_store
from app.qa_chain import build_qa_chain

st.title("📄 RAG PDF Chatbot (Mistral + LangChain)")
uploaded_file = st.file_uploader("Upload a PDF", type="pdf")

if uploaded_file:
    with open(f"data/{uploaded_file.name}", "wb") as f:
        f.write(uploaded_file.read())

    text = extract_text_from_pdf(f"data/{uploaded_file.name}")
    st.success("✅ PDF text extracted")

    chunks = get_chunks(text)
    vector_store = get_vector_store(chunks)
    st.success("✅ Text embedded and stored in vector DB")

    qa_chain = build_qa_chain()

    query = st.text_input("Ask a question about the PDF:")
    if query:
        result = qa_chain.run(query)
        st.write("🔎 Answer:", result)
