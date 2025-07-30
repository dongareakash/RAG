from langchain.chains import RetrievalQA
from langchain.llms import HuggingFaceHub
from langchain.vectorstores import Chroma
from langchain.embeddings import HuggingFaceEmbeddings

def load_vector_store():
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    return Chroma(persist_directory="chroma_db", embedding_function=embeddings)

def build_qa_chain():
    vector_store = load_vector_store()
    retriever = vector_store.as_retriever(search_type="similarity", k=3)

    llm = HuggingFaceHub(
        repo_id="mistralai/Mistral-7B-Instruct-v0.1",
        model_kwargs={"temperature": 0.2, "max_new_tokens": 512}
    )

    chain = RetrievalQA.from_chain_type(llm=llm, retriever=retriever)
    return chain
