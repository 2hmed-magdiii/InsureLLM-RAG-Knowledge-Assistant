from pathlib import Path

from dotenv import load_dotenv

from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma


# ============================================================
# Configuration
# ============================================================

load_dotenv(override=True)

BASE_DIR = Path(__file__).parent

KNOWLEDGE_BASE_PATH = BASE_DIR / "knowledge-base"
DB_PATH = BASE_DIR / "preprocessed_db"

COLLECTION_NAME = "docs"

EMBEDDING_MODEL = "qwen3-embedding:0.6b"

CHUNK_SIZE = 800
CHUNK_OVERLAP = 150


# ============================================================
# 1. Load Documents
# ============================================================

def load_documents():

    print("Loading documents...")

    loader = DirectoryLoader(
        str(KNOWLEDGE_BASE_PATH),
        glob="**/*.md",
        loader_cls=TextLoader,
        loader_kwargs={
            "encoding": "utf-8"
        },
        show_progress=True,
    )

    documents = loader.load()

    print(f"Loaded {len(documents)} documents")

    return documents


# ============================================================
# 2. Split Documents
# ============================================================

def split_documents(documents):

    print("Splitting documents...")

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=[
            "\n\n",
            "\n",
            ". ",
            " ",
            "",
        ],
    )

    chunks = text_splitter.split_documents(
        documents
    )

    print(f"Created {len(chunks)} chunks")

    return chunks


# ============================================================
# 3. Create Embedding Model
# ============================================================

def create_embeddings():

    print(
        f"Loading embedding model: "
        f"{EMBEDDING_MODEL}"
    )

    embeddings = OllamaEmbeddings(
        model=EMBEDDING_MODEL,
        base_url="http://localhost:11434",
    )

    return embeddings


# ============================================================
# 4. Create Vector Store
# ============================================================

def create_vector_store(chunks, embeddings):

    print("Creating vector database...")

    # Delete old database if needed
    if DB_PATH.exists():

        import shutil

        shutil.rmtree(DB_PATH)

        print("Old vector database deleted.")

    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        collection_name=COLLECTION_NAME,
        persist_directory=str(DB_PATH),
    )

    print(
        f"Vectorstore created successfully."
    )

    print(
        f"Total chunks stored: {len(chunks)}"
    )

    return vectorstore


# ============================================================
# Main
# ============================================================

if __name__ == "__main__":

    print("\nStarting ingestion...\n")

    documents = load_documents()

    chunks = split_documents(
        documents
    )

    embeddings = create_embeddings()

    vectorstore = create_vector_store(
        chunks,
        embeddings
    )

    print("\nIngestion complete!")