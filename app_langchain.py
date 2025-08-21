import os
import glob
from typing import List, Tuple

import streamlit as st
from dotenv import load_dotenv

# LangChain imports
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
from langchain_community.vectorstores import FAISS
from langchain_cohere import CohereEmbeddings, ChatCohere
from langchain.prompts import ChatPromptTemplate


BOOKS_DIR = "books"
VECTOR_DIR = "vector_store_langchain"  # separate from the pure-FAISS version
EMBEDDING_MODEL_NAME = "embed-english-v3.0"  # Cohere embedding model
DEFAULT_TOP_K = 5


def ensure_dirs():
    os.makedirs(BOOKS_DIR, exist_ok=True)
    os.makedirs(VECTOR_DIR, exist_ok=True)


def load_books() -> List[Tuple[str, str]]:
    files = sorted(glob.glob(os.path.join(BOOKS_DIR, "*.txt")))
    contents = []
    for fp in files:
        try:
            with open(fp, "r", encoding="utf-8") as f:
                contents.append((os.path.basename(fp), f.read()))
        except Exception as e:
            st.warning(f"Failed to read {fp}: {e}")
    return contents


def get_text_splitter():
    return RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=150,
        separators=["\n\n", "\n", " ", ""],
    )


def get_embeddings():
    # Cohere embeddings via LangChain
    # Note: set COHERE_API_KEY in environment/.env
    return CohereEmbeddings(model=EMBEDDING_MODEL_NAME)


def build_vectorstore(docs: List[Document], embeddings: CohereEmbeddings) -> FAISS:
    vs = FAISS.from_documents(docs, embeddings)
    vs.save_local(VECTOR_DIR)
    return vs


def load_vectorstore(embeddings: CohereEmbeddings):
    try:
        return FAISS.load_local(VECTOR_DIR, embeddings, allow_dangerous_deserialization=True)
    except Exception:
        return None


def rebuild_index(embeddings: CohereEmbeddings):
    books = load_books()
    if not books:
        st.error(f"No books found in '{BOOKS_DIR}/'. Add .txt files and try again.")
        return None

    splitter = get_text_splitter()
    docs: List[Document] = []
    for source, text in books:
        for chunk in splitter.split_text(text):
            docs.append(Document(page_content=chunk, metadata={"source": source}))

    return build_vectorstore(docs, embeddings)


def configure_cohere_model():
    load_dotenv()
    if not os.getenv("COHERE_API_KEY"):
        st.error("Missing COHERE_API_KEY. Set it in a .env file or environment variables.")
        return None
    # Common Cohere chat models: 'command-r', 'command-r-plus'
    return ChatCohere(model="command-r", temperature=0.2)


def generate_answer(llm: ChatCohere, question: str, contexts: List[Document]) -> str:
    context_text = "\n\n".join([f"[Source: {d.metadata.get('source','unknown')}]\n{d.page_content}" for d in contexts])

    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a helpful assistant. Answer strictly using the provided context.\n"
                   "If the answer is not in the context, say: I don't know based on the provided documents."),
        ("human", "Question: {question}\n\nContext:\n{context}\n\nProvide a concise answer with brief [source] citations."),
    ])

    chain = prompt | llm
    resp = chain.invoke({"question": question, "context": context_text})
    try:
        return resp.content.strip()
    except Exception:
        # In case the return type differs between versions
        return str(resp)


def main():
    st.set_page_config(page_title="Simple RAG (LangChain + FAISS + Cohere)", page_icon="📚", layout="wide")
    st.title("📚 Simple RAG (LangChain + FAISS + Cohere)")
    st.caption("Load books from 'books/', retrieve with LangChain FAISS, answer with Cohere.")

    ensure_dirs()

    with st.sidebar:
        st.header("Settings")
        top_k = st.slider("Top-K Chunks", 1, 10, DEFAULT_TOP_K)
        rebuild = st.button("Rebuild Index")
        st.markdown("---")
        st.write("Environment")
        st.code("COHERE_API_KEY=...", language="bash")

    embeddings = get_embeddings()

    vector = load_vectorstore(embeddings)
    if rebuild or vector is None:
        with st.spinner("Building vector store (LangChain FAISS)..."):
            vector = rebuild_index(embeddings)

    question = st.text_input("Ask a question about the books:", placeholder="e.g., What is polymorphism? How do neural networks work?")
    if not question:
        st.stop()

    if vector is None:
        st.error("Vector store not ready. Please rebuild after adding books.")
        st.stop()

    with st.spinner("Retrieving relevant chunks..."):
        retriever = vector.as_retriever(search_type="similarity", search_kwargs={"k": top_k})
        docs = retriever.invoke(question)

    if not docs:
        st.warning("No results found.")
        st.stop()

    st.subheader("Top Retrieved Chunks")
    for i, d in enumerate(docs, 1):
        src = d.metadata.get("source", "unknown")
        with st.expander(f"{i}. {src}"):
            st.write(d.page_content)

    llm = configure_cohere_model()
    if llm is None:
        st.stop()

    with st.spinner("Generating answer with Cohere (LangChain)..."):
        answer = generate_answer(llm, question, docs)

    st.subheader("Answer")
    st.write(answer)


if __name__ == "__main__":
    main()
