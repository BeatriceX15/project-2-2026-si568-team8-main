import os
import re
import streamlit as st
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_openai import ChatOpenAI
# step 2
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
# step 3
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma

# Load environment variables
load_dotenv()

# ==========================================
# STEP 1: LOAD DOCUMENTS
# ==========================================


def load_school_policies(pdf_directory):
    """
    Load all PDF files from a directory and convert them to text documents.
    """
    documents = []

    if not os.path.exists(pdf_directory):
        st.error(
            f"Error: Could not find the directory '{pdf_directory}'. Make sure your folders are set up correctly!")
        return documents

    for filename in os.listdir(pdf_directory):
        if filename.endswith(".pdf"):
            filepath = os.path.join(pdf_directory, filename)
            st.text(f"Loading: {filename}...")
            loader = PyPDFLoader(filepath)
            documents.extend(loader.load())

    for doc in documents:
        doc.page_content = re.sub(r' {2,}', ' ', doc.page_content)
        doc.page_content = re.sub(r'\n\s*\n', '\n', doc.page_content)

    st.success(
        f"Successfully loaded {len(documents)} total pages from the UMSI Handbooks!")
    return documents

# ==========================================
# STEP 2: CHUNK THE TEXT
# ==========================================


def chunk_documents(documents, chunk_size=1500, chunk_overlap=200):
    """
    Split the loaded documents into smaller chunks for better embedding and retrieval performance.
    Args:
        documents (list): List of document objects loaded from PDFs.
        chunk_size (int): Number of characters in each chunk.
        chunk_overlap (int): Number of characters to overlap between chunks.
    """
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )
    all_chunks = []
    for doc in documents:
        chunks = text_splitter.split_text(doc.page_content)
        for chunk in chunks:
            all_chunks.append(
                Document(page_content=chunk, metadata=doc.metadata))

    return all_chunks

# ==========================================
# STEP 3: EMBED & STORE
# ==========================================


def build_vector_store(chunks, persist_directory="data/chroma_db"):
    """
    Embed document chunks using OpenAI embeddings and store them in a Chroma vector database.

    Args:
        chunks (list): List of Document objects to embed.
        persist_directory (str): Path to persist the Chroma database.

    Returns:
        Chroma: The vector store instance, or None if chunks are empty.
    """
    if not chunks:
        st.error("No chunks provided. Please load and chunk documents first.")
        return None

    embeddings = OpenAIEmbeddings(
        model=os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small"),
        base_url=os.getenv("OPENAI_BASE_URL"),
    )

    vector_store = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=persist_directory,
    )

    st.success(
        f"Vector store built with {len(chunks)} chunks, saved to '{persist_directory}'.")
    return vector_store

# ==========================================
# STEP 4: RETRIEVAL LOGIC
# ==========================================


def retrieve_chunks(vector_store, test_question, k=8):
    """
    TODO: Turn the Chroma database into a retriever and test a question.
    """
    if not vector_store:
        return "No vector store found. Please build the vector store first."
    if not test_question:
        return "Please provide a question."

    retriever = vector_store.as_retriever(
        search_type="similarity",
        search_kwargs={"k": k}
    )
    retrieved_chunks = retriever.invoke(test_question)

   

    return retrieved_chunks


# ==========================================
# STEP 5: LLM GENERATION
# ==========================================


def generate_answer(retrieved_chunks, user_question):
    """
    Integrate the LLM (ChatOpenAI) for question answering.

    Args:
        retrieved_chunks (list): List of Document objects with relevant content.
        user_question (str): The user's question.

    Returns:
        str: The generated answer from the LLM.
    """
    if not retrieved_chunks:
        return "No relevant information found in the documents to answer your question."

    if not user_question:
        return "Please provide a question."

    # Create LLM instance
    llm = ChatOpenAI(
        model=os.getenv("OPENAI_MODEL", "gpt-4"),
        base_url=os.getenv("OPENAI_BASE_URL"),
        temperature=0,
    )

    # Combine chunks into a single string
    context = "\n\n".join([chunk.page_content for chunk in retrieved_chunks])

    # LLM prompt
    prompt = f"""You are a helpful assistant for UMSI students. Answer the following question using ONLY the provided documents.
If the answer is not in the documents, say "I don't have information about this in the available documents." Do not try to make up an answer. 

Documents:
{context}

Question: {user_question}

Answer:"""

    # Call LLM
    response = llm.invoke(prompt)

    # Return answer text
    return response.content

# ==========================================
# SYSTEM INITIALIZATION
# ==========================================


@st.cache_resource
def initialize_system(pdf_folder_path, chroma_dir, chunk_size, chunk_overlap):
    """
    Load existing Chroma DB if present, otherwise build it.
    """
    embeddings = OpenAIEmbeddings(
        model=os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small"),
        base_url=os.getenv("OPENAI_BASE_URL"),
    )

    if os.path.exists(chroma_dir) and os.listdir(chroma_dir):
        return Chroma(
            persist_directory=chroma_dir,
            embedding_function=embeddings
        )

    documents = load_school_policies(pdf_folder_path)
    if not documents:
        return None

    chunks = chunk_documents(
        documents,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )

    return build_vector_store(chunks, persist_directory=chroma_dir)


# ==========================================
# STREAMLIT UI
# ==========================================

def launch_ui():
    st.set_page_config(page_title="UMSI Policy Assistant", page_icon="〽️")
    st.title("〽️ UMSI Policy FAQ Assistant")
    st.write("Ask a question about MSI degree requirements, registration, internships, or program policies.")

    # Sidebar settings
    st.sidebar.header("Settings")
    pdf_folder_path = st.sidebar.text_input("PDF folder path", "data/raw_pdfs")
    chroma_dir = st.sidebar.text_input("Chroma DB path", "data/chroma_db")
    k = st.sidebar.slider("Number of retrieved chunks (k)", 1, 15, 8)
    chunk_size = st.sidebar.slider("Chunk size", 500, 3000, 1500, step=100)
    chunk_overlap = st.sidebar.slider("Chunk overlap", 0, 500, 200, step=25)

    # Build / load vector store
    with st.spinner("Initializing document system..."):
        vector_store = initialize_system(
            pdf_folder_path=pdf_folder_path,
            chroma_dir=chroma_dir,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap
        )

    if vector_store is None:
        st.stop()

    st.success("System ready.")

    # Question input
    user_input = st.text_input("Student Question:")

    if st.button("Ask"):
        if not user_input.strip():
            st.warning("Please enter a question first.")
            return

        with st.spinner("Searching documents and generating answer..."):
            retrieved_chunks = retrieve_chunks(vector_store, user_input, k=k)
            answer = generate_answer(retrieved_chunks, user_input)

        # Answer
        st.subheader("Answer")
        st.info(answer)

        # Sources
        st.subheader("Sources Used")
        if retrieved_chunks:
            for i, doc in enumerate(retrieved_chunks):
                source_file = doc.metadata.get("source", "Unknown")
                source_file = os.path.basename(source_file)
                page_num = doc.metadata.get("page", "Unknown")

                with st.expander(f"[{i+1}] {source_file} (Page {page_num})"):
                    st.write(doc.page_content)
                    st.json(doc.metadata)
        else:
            st.write("No source chunks were retrieved.")


def test_pipeline():
    PDF_FOLDER_PATH = "data/raw_pdfs"
    TEST_QUESTION = "What are the detailed degree requirements for MSI Big Data Analytics track?"

    CHROMA_DIR = "data/chroma_db"
    embeddings = OpenAIEmbeddings(
        model=os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small"),
        base_url=os.getenv("OPENAI_BASE_URL"),
    )

    if os.path.exists(CHROMA_DIR) and os.listdir(CHROMA_DIR):
        print("=" * 40)
        print("Steps 1-3: Loading existing vector store (skipping rebuild)...")
        print("=" * 40)
        vector_store = Chroma(persist_directory=CHROMA_DIR,
                              embedding_function=embeddings)
        print(f"  Loaded vector store from '{CHROMA_DIR}'.\n")
    else:
        print("=" * 40)
        print("Step 1: Loading documents...")
        print("=" * 40)
        documents = []
        for filename in sorted(os.listdir(PDF_FOLDER_PATH)):
            if filename.endswith(".pdf"):
                print(f"  Loading: {filename}")
                loader = PyPDFLoader(os.path.join(PDF_FOLDER_PATH, filename))
                documents.extend(loader.load())
        for doc in documents:
            doc.page_content = re.sub(r' {2,}', ' ', doc.page_content)
            doc.page_content = re.sub(r'\n\s*\n', '\n', doc.page_content)
        print(f"  Loaded {len(documents)} pages.\n")

        print("=" * 40)
        print("Step 2: Chunking documents...")
        print("=" * 40)
        chunks = chunk_documents(documents, chunk_size=1500, chunk_overlap=200)
        print(f"  Created {len(chunks)} chunks.\n")

        print("=" * 40)
        print("Step 3: Building vector store...")
        print("=" * 40)
        vector_store = Chroma.from_documents(
            documents=chunks,
            embedding=embeddings,
            persist_directory=CHROMA_DIR,
        )
        print(f"  Vector store built with {len(chunks)} chunks.\n")

    print("=" * 40)
    print(f"Step 4: Retrieving for question: '{TEST_QUESTION}'")
    print("=" * 40)
    retriever = vector_store.as_retriever(
        search_type="similarity", search_kwargs={"k": 8})
    retrieved_chunks = retriever.invoke(TEST_QUESTION)
    print(f"  Retrieved {len(retrieved_chunks)} chunks.")
    for i, chunk in enumerate(retrieved_chunks):
        print(f"  Chunk {i+1}: {chunk.page_content[:100]}...")
    print()

    print("=" * 40)
    print("Step 5: Generating answer...")
    print("=" * 40)
    answer = generate_answer(retrieved_chunks, TEST_QUESTION)
    print(f"\nAnswer:\n{answer}")


if __name__ == "__main__":
    import sys
    launch_ui()
