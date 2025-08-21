# Simple RAG (Books + FAISS + Cohere)

A minimal Retrieval-Augmented Generation demo over the 3 text files under `books/`.
It uses:
- Cohere embeddings (`embed-english-v3.0`) via LangChain
- FAISS for vector search (LangChain vector store)
- Cohere chat model (`command-r`) for generation via LangChain
- Streamlit UI

## Setup

1) Python 3.9–3.11 recommended.

2) Install dependencies:
```powershell
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

3) Create `.env` and set your Cohere API key:
```powershell
copy .env.example .env   # Windows
# then edit .env and set COHERE_API_KEY
```

4) Run the app:
```powershell
streamlit run app_langchain.py
```

5) In the UI:
- Use the sidebar to rebuild the FAISS index if you add/modify files in `books/`.
- Ask a question about the books.

## Notes
- Vector store files are saved to `vector_store_langchain/`. Delete it or click "Rebuild Index" to refresh.
- If FAISS install fails on Windows, ensure you are using `faiss-cpu` and a compatible Python version. You can also try a prebuilt wheel.
