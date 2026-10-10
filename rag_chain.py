import os
import io
import pdfplumber
from typing import List, Tuple, Dict, Any

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough


def extract_documents_from_pdfs(uploaded_files) -> List[Document]:
    """
    Extracts text from uploaded PDF files and converts them into LangChain Document objects
    with metadata (source filename, page number).
    """
    documents = []
    for file in uploaded_files:
        filename = getattr(file, "name", "document.pdf")
        # Read file bytes into pdfplumber
        if hasattr(file, "read"):
            file_bytes = file.read()
            file.seek(0)
            pdf_file = io.BytesIO(file_bytes)
        else:
            pdf_file = file

        with pdfplumber.open(pdf_file) as pdf:
            for page_idx, page in enumerate(pdf.pages, start=1):
                page_text = page.extract_text()
                if page_text and page_text.strip():
                    documents.append(
                        Document(
                            page_content=page_text.strip(),
                            metadata={
                                "source": filename,
                                "page": page_idx,
                            },
                        )
                    )
    return documents


def split_documents(
    documents: List[Document], chunk_size: int = 1000, chunk_overlap: int = 200
) -> List[Document]:
    """
    Splits documents into overlapping chunks for semantic retrieval.
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", " ", ""],
    )
    return splitter.split_documents(documents)


def create_vector_store(
    chunks: List[Document], api_key: str = None
) -> FAISS:
    """
    Builds a FAISS vector index using OpenAI embeddings.
    """
    embeddings = OpenAIEmbeddings(
        openai_api_key=api_key or os.getenv("OPENAI_API_KEY"),
        model="text-embedding-3-small",
    )
    vector_store = FAISS.from_documents(chunks, embeddings)
    return vector_store


def format_retrieved_docs(docs: List[Document]) -> str:
    """
    Formats retrieved documents into context with source citations.
    """
    formatted = []
    for i, doc in enumerate(docs, start=1):
        source = doc.metadata.get("source", "Unknown Document")
        page = doc.metadata.get("page", "?")
        formatted.append(f"[Source {i}: {source} (Page {page})]\n{doc.page_content}")
    return "\n\n---\n\n".join(formatted)


def run_rag_pipeline(
    vector_store: FAISS,
    query: str,
    model_name: str = "gpt-3.5-turbo",
    temperature: float = 0.3,
    api_key: str = None,
    k: int = 4,
) -> Tuple[str, List[Document]]:
    """
    Executes the LangChain RAG pipeline: retrieves top-k documents and generates a response.
    Returns: (response_text, retrieved_documents)
    """
    retriever = vector_store.as_retriever(search_kwargs={"k": k})
    retrieved_docs = retriever.invoke(query)

    if not retrieved_docs:
        return "No relevant context found in the uploaded documents to answer your question.", []

    formatted_context = format_retrieved_docs(retrieved_docs)

    prompt = ChatPromptTemplate.from_messages([
        (
            "system",
            "You are an expert AI document research assistant. "
            "Answer the user's question based strictly on the provided context below. "
            "If the answer cannot be found in the context, politely explain that the document does not contain this information. "
            "Cite the specific document sources and page numbers whenever applicable."
        ),
        (
            "human",
            "Context:\n{context}\n\nQuestion:\n{question}\n\nDetailed Answer:"
        )
    ])

    llm = ChatOpenAI(
        model=model_name,
        temperature=temperature,
        openai_api_key=api_key or os.getenv("OPENAI_API_KEY"),
    )

    chain = prompt | llm | StrOutputParser()
    response = chain.invoke({"context": formatted_context, "question": query})

    return response, retrieved_docs
