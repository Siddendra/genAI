import os
os.environ.setdefault(
    "USER_AGENT",
    "RockyBot/1.0 (contact: siddendra@example.com)"
)
import streamlit as st
import pickle
import time
from dotenv import load_dotenv

from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.document_loaders import WebBaseLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA

# -----------------------------
# Setup
# -----------------------------
load_dotenv()  # pulls OPENAI_API_KEY from .env if present

st.set_page_config(page_title="RockyBot: News Research Tool", page_icon="📈")
st.title("RockyBot: News Research Tool 📈")
st.sidebar.title("News Article URLs")

# Storage directory for FAISS index
INDEX_DIR = "faiss_store_openai"

# Small helper to guard key availability
OPENAI_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_KEY:
    st.sidebar.warning("⚠️ OPENAI_API_KEY not found. Add it to your .env or environment.")


# Model/embedding instances (created lazily)
def get_llm():
    # Pick a capable, cost-effective chat model; change as you wish
    return ChatOpenAI(model="gpt-4o-mini", temperature=0.3)


def get_embeddings():
    return OpenAIEmbeddings()


# -----------------------------
# Sidebar: URL inputs + Process
# -----------------------------
urls = []
for i in range(3):
    url = st.sidebar.text_input(f"URL {i + 1}", key=f"url_{i}")
    if url.strip():
        urls.append(url.strip())

process_url_clicked = st.sidebar.button("Process URLs")

status_placeholder = st.empty()

# -----------------------------
# Build / Update the Vector Index
# -----------------------------
if process_url_clicked:
    if not urls:
        st.sidebar.error("Please enter at least one valid URL.")
    elif not OPENAI_KEY:
        st.sidebar.error("Missing OPENAI_API_KEY. Add it to .env or your environment.")
    else:
        try:
            status_placeholder.text("🔗 Loading articles...")
            loaders = [WebBaseLoader(url) for url in urls]
            docs = []
            for loader in loaders:
                docs.extend(loader.load())

            status_placeholder.text("✂️ Splitting text into chunks...")
            splitter = RecursiveCharacterTextSplitter(
                chunk_size=1000,
                chunk_overlap=150
            )
            split_docs = splitter.split_documents(docs)

            status_placeholder.text("🧠 Creating embeddings & FAISS index...")
            embeddings = get_embeddings()
            vs = FAISS.from_documents(split_docs, embeddings)

            # Save FAISS index safely
            vs.save_local(INDEX_DIR)
            status_placeholder.text("✅ Index built & saved!")
            time.sleep(1)
            status_placeholder.empty()
        except Exception as e:
            status_placeholder.empty()
            st.error(f"Failed to process URLs: {e}")

# -----------------------------
# Query Section
# -----------------------------
st.subheader("Ask a question about the articles")
query = st.text_input("Question:", placeholder="e.g., What are the key takeaways and where do they disagree?")

if query:
    if not os.path.exists(INDEX_DIR):
        st.error("No index found. Add URLs and click 'Process URLs' first.")
    elif not OPENAI_KEY:
        st.error("Missing OPENAI_API_KEY. Add it to .env or your environment.")
    else:
        try:
            # Load index
            embeddings = get_embeddings()
            vectorstore = FAISS.load_local(
                INDEX_DIR,
                embeddings,
                allow_dangerous_deserialization=True  # required for FAISS load_local
            )

            retriever = vectorstore.as_retriever(search_kwargs={"k": 4})
            llm = get_llm()

            # RetrievalQA returns answer + (optionally) source documents
            qa = RetrievalQA.from_chain_type(
                llm=llm,
                chain_type="stuff",
                retriever=retriever,
                return_source_documents=True
            )

            with st.spinner("Thinking..."):
                result = qa({"query": query})

            st.header("Answer")
            st.write(result.get("result", ""))

            # Show sources (URLs) if available
            src_docs = result.get("source_documents", []) or []
            if src_docs:
                st.subheader("Sources")
                seen = set()
                for d in src_docs:
                    src = d.metadata.get("source") or d.metadata.get("url") or ""
                    if src and src not in seen:
                        seen.add(src)
                        st.write(f"- {src}")
        except Exception as e:
            st.error(f"Query failed: {e}")
