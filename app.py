import os
import streamlit as st
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.chains import RetrievalQA
from langchain_groq import ChatGroq  # 👈 using Groq instead of OpenAI

# ----------------------------
# CONFIG
# ----------------------------

os.environ["GROQ_API_KEY"] = "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"
os.environ["HF_TOKEN"] = "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"

# Embedding model
EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
# LLM model
LLM_MODEL = "llama-3.3-70b-versatile"
# Knowledge file
FILE_PATH = "knowledge.txt"

# ----------------------------
# LOAD EMBEDDINGS
# ----------------------------
@st.cache_resource
def load_vectordb():
    embeddings = HuggingFaceEmbeddings(model_name=EMBED_MODEL)

    if not os.path.exists(FILE_PATH):
        raise FileNotFoundError("📂 knowledge.txt not found. Please create it!")

    with open(FILE_PATH, "r", encoding="utf-8") as f:
        content = f.read().strip()

    if not content:
        raise ValueError("📂 knowledge.txt is empty. Add some text first ✍️")

    # 👇 split into small chunks for embeddings
    texts = content.split(". ")
    vectordb = Chroma.from_texts(texts, embeddings)
    return vectordb


vectordb = load_vectordb()

# ----------------------------
# SET UP RETRIEVER + LLM
# ----------------------------
retriever = vectordb.as_retriever(search_kwargs={"k": 3})
llm = ChatGroq(model=LLM_MODEL, temperature=0)

qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=retriever,
    return_source_documents=True
)

# ----------------------------
# STREAMLIT UI
# ----------------------------
st.title("💡 Local Knowledge Q&A Bot 🤖")

st.write("Ask me questions based on your `knowledge.txt` file 📄")

user_q = st.text_input("❓ Enter your question:")

if user_q:
    with st.spinner("🤔 Thinking..."):
        response = qa_chain.invoke(user_q)

    st.subheader("✅ Answer:")
    st.write(response["result"])

    st.subheader("📚 Sources:")
    for doc in response["source_documents"]:
        st.write(f"- {doc.page_content}")