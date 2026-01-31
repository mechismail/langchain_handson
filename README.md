# LangChain RAG with Hugging Face (FLAN-T5)

This project demonstrates a **Retrieval-Augmented Generation (RAG)** pipeline using **LangChain**, **Hugging Face Transformers**, and **FAISS**, running fully **locally** (no OpenAI API required).

It answers user questions by retrieving relevant information from a local text file and generating grounded responses using a Hugging Face LLM.

---

## ✨ Features

- Local LLM using **google/flan-t5-base**
- Document-based question answering (RAG)
- Vector search with **FAISS**
- Sentence embeddings using **Sentence Transformers**
- Optional web search tool via **Tavily**
- Clean project structure, GitHub-ready

---

## 🧠 How It Works (RAG Flow)

1. Load documents from `sample.txt`
2. Convert text into embeddings
3. Store embeddings in FAISS vector database
4. Retrieve relevant chunks for a query
5. Generate a grounded answer using the LLM

---

## 📁 Project Structure

```text
langchain_handson/
│
├── main.py                 # Main application
├── sample.txt              # Knowledge source for RAG
├── requirements.txt        # Clean dependency list
├── requirements-lock.txt   # Exact environment snapshot
├── .gitignore              # Ignored files (env, venv, cache)
└── README.md               # Project documentation

```
