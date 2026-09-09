import os
import time
import streamlit as st
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_community.document_loaders import UnstructuredURLLoader, PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.chains import RetrievalQAWithSourcesChain

load_dotenv()

st.set_page_config(page_title="PrajwalGPT", layout="wide")

GROQ_API_KEY = os.environ.get("GROQ_API_KEY")

MODEL_OPTIONS = {
    "GPT OSS 120B (Best Quality)": "openai/gpt-oss-120b",
    "GPT OSS 20B (Fast)": "openai/gpt-oss-20b",
}

if "messages" not in st.session_state:
    st.session_state.messages = []
if "vectorstore" not in st.session_state:
    st.session_state.vectorstore = None
if "processed_sources" not in st.session_state:
    st.session_state.processed_sources = []


@st.cache_resource
def get_embeddings():
    return HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")


def build_llm(model_name, temperature):
    return ChatGroq(groq_api_key=GROQ_API_KEY, model_name=model_name, temperature=temperature, streaming=True)


def to_lc_messages(messages):
    lc = [SystemMessage(content="You are a helpful, friendly and concise AI assistant.")]
    for m in messages:
        if m["role"] == "user":
            lc.append(HumanMessage(content=m["content"]))
        else:
            lc.append(AIMessage(content=m["content"]))
    return lc


def process_sources(urls, uploaded_files):
    docs = []
    valid_urls = [u.strip() for u in urls if u.strip()]
    if valid_urls:
        loader = UnstructuredURLLoader(urls=valid_urls)
        docs.extend(loader.load())
    for f in uploaded_files or []:
        os.makedirs("temp_uploads", exist_ok=True)
        path = os.path.join("temp_uploads", f.name)
        with open(path, "wb") as out:
            out.write(f.getbuffer())
        loader = PyPDFLoader(path)
        docs.extend(loader.load())
    if not docs:
        return None
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
    chunks = splitter.split_documents(docs)
    embeddings = get_embeddings()
    return FAISS.from_documents(chunks, embeddings)


with st.sidebar:
    st.title("PrajwalGPT")
    st.caption("Free RAG Chatbot powered by Groq + local embeddings")

    st.subheader("Model Settings")
    selected_model_label = st.selectbox("Choose Model", list(MODEL_OPTIONS.keys()))
    temperature = st.slider("Creativity", 0.0, 1.0, 0.3, 0.1)

    st.subheader("Add Knowledge Base")
    url_count = st.number_input("Number of URLs", min_value=0, max_value=5, value=1)
    urls = [st.text_input(f"URL {i + 1}", key=f"url_{i}") for i in range(url_count)]
    uploaded_files = st.file_uploader("Upload PDFs", type=["pdf"], accept_multiple_files=True)

    if st.button("Process Sources", use_container_width=True):
        with st.spinner("Reading and embedding your documents..."):
            vs = process_sources(urls, uploaded_files)
            if vs:
                st.session_state.vectorstore = vs
                st.session_state.processed_sources = [u for u in urls if u.strip()] + [f.name for f in uploaded_files or []]
                st.success(f"Processed {len(st.session_state.processed_sources)} source(s)!")
            else:
                st.warning("No valid sources provided.")

    if st.session_state.processed_sources:
        st.subheader("Active Sources")
        for s in st.session_state.processed_sources:
            st.text(s)

    col1, col2 = st.columns(2)
    with col1:
        if st.button("Clear KB", use_container_width=True):
            st.session_state.vectorstore = None
            st.session_state.processed_sources = []
            st.rerun()
    with col2:
        if st.button("Clear Chat", use_container_width=True):
            st.session_state.messages = []
            st.rerun()

    if st.session_state.messages:
        chat_text = "\n\n".join([f"{m['role'].upper()}: {m['content']}" for m in st.session_state.messages])
        st.download_button("Download Chat", chat_text, file_name="chat_history.txt", use_container_width=True)

st.title("Chat with PrajwalGPT")
st.caption(f"Model: {selected_model_label} | Knowledge Base: {'Active' if st.session_state.vectorstore else 'Not Set'}")

if not GROQ_API_KEY:
    st.warning("GROQ_API_KEY not found in .env file. Get a free key at https://console.groq.com and add it before chatting.")

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if message.get("sources"):
            with st.expander("Sources"):
                st.markdown(message["sources"])

if prompt := st.chat_input("Ask me anything..."):
    if not GROQ_API_KEY:
        st.error("Please set GROQ_API_KEY in your .env file first.")
        st.stop()

    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        placeholder = st.empty()
        full_response = ""
        sources_text = ""
        model_name = MODEL_OPTIONS[selected_model_label]
        llm = build_llm(model_name, temperature)

        if st.session_state.vectorstore:
            chain = RetrievalQAWithSourcesChain.from_llm(llm=llm, retriever=st.session_state.vectorstore.as_retriever())
            result = chain({"question": prompt}, return_only_outputs=True)
            full_response = result.get("answer", "")
            sources_text = result.get("sources", "")
            placeholder.markdown(full_response)
        else:
            lc_history = to_lc_messages(st.session_state.messages)
            for chunk in llm.stream(lc_history):
                full_response += chunk.content
                placeholder.markdown(full_response + "▌")
                time.sleep(0.01)
            placeholder.markdown(full_response)

    st.session_state.messages.append({"role": "assistant", "content": full_response, "sources": sources_text})
