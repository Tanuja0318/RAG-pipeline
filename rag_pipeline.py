# rag_pipeline.py — Retrieves relevant chunks and generates answers using LLM

import os
from dotenv import load_dotenv
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

load_dotenv()

CHROMA_DIR = os.getenv("CHROMA_PERSIST_DIR", "./chroma_db")


def get_vectorstore():
    """Load the existing ChromaDB vectorstore from disk"""
    embeddings = HuggingFaceEmbeddings(
        model_name="all-MiniLM-L6-v2",
        model_kwargs={"device": "cpu"}
    )
    vectorstore = Chroma(
        persist_directory=CHROMA_DIR,
        embedding_function=embeddings,
        collection_name="rag_docs"
    )
    return vectorstore


def build_rag_chain():
    """Build the full RAG chain: retrieve → prompt → LLM → answer"""

    vectorstore = get_vectorstore()

    # Retriever: fetch top 4 most similar chunks for every query
    retriever = vectorstore.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 4}
    )

    # LLM: GPT-4o-mini is cheap and very good for Q&A
    llm = ChatGroq(
        model="llama-3.3-70b-versatile",
        temperature=0.2,
        api_key=os.getenv("GROQ_API_KEY")
    )

    # Prompt template: tells LLM to only use retrieved context
    prompt = ChatPromptTemplate.from_template("""
You are a helpful assistant. Answer the question using ONLY the context below.
If the answer isn't in the context, say "I don't have enough information to answer that."

Context:
{context}

Question: {question}

Answer:""")

    def format_docs(docs):
        """Join all retrieved chunks into one context string"""
        return "\n\n---\n\n".join(doc.page_content for doc in docs)

    # LangChain Expression Language (LCEL) chain
    chain = (
        {
            "context": retriever | format_docs,  # Retrieve + format
            "question": RunnablePassthrough()     # Pass question unchanged
        }
        | prompt          # Fill prompt template
        | llm             # Send to LLM
        | StrOutputParser() # Extract text from response
    )

    return chain, retriever


# Initialize once when module loads (avoids reloading on every request)
rag_chain, retriever = build_rag_chain()


def query_rag(question: str) -> dict:
    """Run the RAG pipeline and return answer + source chunks"""

    # Get answer from chain
    answer = rag_chain.invoke(question)

    # Also retrieve source documents to show the user what was used
    source_docs = retriever.invoke(question)
    sources = [
        {
            "content": doc.page_content[:200],          # First 200 chars
            "source": doc.metadata.get("source", "Unknown"),
            "page": doc.metadata.get("page", "N/A")
        }
        for doc in source_docs
    ]

    return {
        "answer": answer,
        "sources": sources
    }