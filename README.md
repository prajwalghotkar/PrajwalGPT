# PrajwalGPT

A free, open-source Retrieval-Augmented Generation (RAG) chatbot built with Streamlit, LangChain, Groq, and local embeddings. Chat normally, or feed it URLs and PDFs so it can answer questions grounded in your own documents — with zero paid API costs.

## Overview

PrajwalGPT bridges the gap between raw documents and conversational AI. Upload a PDF or paste a URL, and the app builds a searchable knowledge base from it. Ask questions, and the assistant retrieves the most relevant content and answers with cited sources — instead of relying purely on the model's training data.

If no knowledge base is loaded, it works as a regular streaming chat assistant.

## Features

- **Free LLM inference** via Groq (Llama 3.3 70B, Llama 3.1 8B, Gemma2 9B)
- **Local, free embeddings** using `sentence-transformers` — no embedding API costs, runs on CPU
- **Retrieval-Augmented Generation** — ask questions grounded in your own PDFs or web pages
- **Source citations** — every RAG answer shows which document/URL it came from
- **Multi-source ingestion** — process multiple URLs and multiple PDFs together
- **Dual mode** — automatically falls back to plain conversational chat when no knowledge base is set
- **Adjustable creativity** via a temperature slider
- **Persistent chat session** with full history
- **One-click reset** for knowledge base or chat history
- **Export conversation** to a `.txt` file

## Architecture

```
User Input (URL / PDF)
        |
        v
Document Loader (UnstructuredURLLoader / PyPDFLoader)
        |
        v
Text Splitter (chunked with overlap)
        |
        v
Local Embedding Model (sentence-transformers/all-MiniLM-L6-v2)
        |
        v
FAISS Vector Store (in-memory similarity search)
        |
        v
User Question --> Retriever --> Relevant Chunks
        |
        v
Groq LLM (Llama 3.3 / 3.1 / Gemma2)
        |
        v
Answer + Cited Sources (streamed to the UI)
```

## Tech Stack

| Component | Choice | Why |
|---|---|---|
| UI | Streamlit | Fast to build, great for chat interfaces |
| LLM Provider | Groq | Free tier, extremely low latency, open-weight models |
| Embeddings | sentence-transformers (local) | No API cost, runs offline, privacy-friendly |
| Vector Store | FAISS | Lightweight, in-memory, no hosting required |
| Orchestration | LangChain | Document loaders, text splitting, RAG chains |

## Getting Started

### Prerequisites

- Python 3.10
- Conda (recommended) or venv
- A free Groq API key from [console.groq.com](https://console.groq.com)

### Installation

```bash
conda create -n env_langchain1 python=3.10
conda activate env_langchain1
pip install -r requirements.txt
```

### Configuration

Create a `.env` file in the project root:

```
GROQ_API_KEY=your_free_groq_key_here
```

Never commit your `.env` file. It's recommended to add it to `.gitignore`.

### Run the App

```bash
streamlit run chatgpt_like_app.py
```

The app will open in your browser at `http://localhost:8501`.

## Usage

1. **Plain chat**: Just start typing in the chat box — no setup required.
2. **RAG mode**:
   - In the sidebar, add one or more URLs and/or upload PDF files.
   - Click **Process Sources** to build the knowledge base.
   - Ask questions — answers will be grounded in your documents, with sources shown in an expandable section.
3. Use **Clear KB** to remove the current knowledge base, or **Clear Chat** to reset the conversation.
4. Use **Download Chat** to export the full conversation as a text file.

## Project Structure

```
prajwalgpt/
├── chatgpt_like_app.py     # Main Streamlit application
├── requirements.txt        # Python dependencies
├── .env                    # API key (not committed)
└── temp_uploads/           # Auto-created, temporary storage for uploaded PDFs
```

## Roadmap

- [ ] Persist the FAISS index to disk between sessions
- [ ] Support additional file formats (CSV, DOCX, TXT)
- [ ] Conversation summarization for long chat histories
- [ ] Multi-user support with isolated knowledge bases

## License

This project is open source and available under the MIT License.

## Acknowledgements

Built with [Streamlit](https://streamlit.io), [LangChain](https://www.langchain.com), [Groq](https://groq.com), and [sentence-transformers](https://www.sbert.net).
