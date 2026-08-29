from pathlib import Path
import json
import re

from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings, ChatOllama
from langchain_core.documents import Document
from langchain_core.messages import SystemMessage, HumanMessage


# ============================================================
# Configuration
# ============================================================

BASE_DIR = Path(__file__).parent

DB_PATH = BASE_DIR / "preprocessed_db"

COLLECTION_NAME = "docs"

EMBEDDING_MODEL = "qwen3-embedding:0.6b"

RERANKER_MODEL = "qwen3:4b"

OLLAMA_BASE_URL = "http://localhost:11434"

# Retrieve more candidates first
RETRIEVAL_K = 20

# Keep only the best chunks after reranking
FINAL_K = 10


# ============================================================
# Lazy-loaded singletons
# ============================================================

_embeddings = None
_vectorstore = None
_reranker_llm = None


def get_embeddings():
    global _embeddings
    if _embeddings is None:
        _embeddings = OllamaEmbeddings(
            model=EMBEDDING_MODEL,
            base_url=OLLAMA_BASE_URL,
        )
    return _embeddings


def get_vectorstore():
    global _vectorstore
    if _vectorstore is None:
        _vectorstore = Chroma(
            collection_name=COLLECTION_NAME,
            persist_directory=str(DB_PATH),
            embedding_function=get_embeddings(),
        )
    return _vectorstore


def get_reranker_llm():
    global _reranker_llm
    if _reranker_llm is None:
        _reranker_llm = ChatOllama(
            model=RERANKER_MODEL,
            temperature=0,
            base_url=OLLAMA_BASE_URL,
        )
    return _reranker_llm


# ============================================================
# Query Rewriting
# ============================================================

def rewrite_query(
    question: str,
    history: list[dict] | None = None,
) -> str:
    """
    Rewrite the user's question into a more precise
    knowledge-base search query.
    """

    if history is None:
        history = []

    history_text = "\n".join(
        f"{message['role']}: {message['content']}"
        for message in history
    )

    system_text = """You are a query rewriting assistant for an insurance
company knowledge base.

Your job is to rewrite the user's question into a
short, precise and self-contained search query.

Rules:
1. Preserve the original meaning.
2. If the question depends on conversation history,
   include the necessary information from the history.
3. Make the query specific enough to retrieve the
   correct knowledge-base documents.
4. If the question is already clear and specific,
   keep it almost unchanged.
5. Return ONLY the rewritten query."""

    user_text = f"""Conversation history:
{history_text}

Current user question:
{question}

Rewritten query:"""

    try:
        response = get_reranker_llm().invoke(
            [
                SystemMessage(content=system_text),
                HumanMessage(content=user_text),
            ]
        )
        rewritten_query = response.content.strip()
    except Exception as e:
        print(f"[rewrite_query] Error: {e}")
        rewritten_query = question

    return rewritten_query


# ============================================================
# Basic Retrieval
# ============================================================

def fetch_context_unranked(
    question: str,
) -> list[Document]:
    """
    Retrieve the most semantically similar chunks
    from the vector database.
    """

    retriever = get_vectorstore().as_retriever(
        search_kwargs={
            "k": RETRIEVAL_K
        }
    )

    documents = retriever.invoke(question)

    return documents


# ============================================================
# Merge Retrieved Chunks
# ============================================================

def merge_chunks(
    chunks1: list[Document],
    chunks2: list[Document],
) -> list[Document]:
    """
    Merge results from multiple retrieval queries
    while removing duplicate chunks.
    """

    merged = []

    seen = set()

    for chunk in chunks1 + chunks2:

        source = chunk.metadata.get(
            "source",
            ""
        )

        key = (
            source,
            chunk.page_content,
        )

        if key not in seen:

            seen.add(key)

            merged.append(chunk)

    return merged


# ============================================================
# Reranking
# ============================================================

def rerank(
    question: str,
    chunks: list[Document],
) -> list[Document]:
    """
    Use an LLM to rerank retrieved chunks according
    to their relevance to the user's question.
    """

    if not chunks:
        return []

    # --------------------------------------------------------
    # Build chunks prompt
    # --------------------------------------------------------

    chunks_text = ""

    for index, chunk in enumerate(
        chunks,
        start=1
    ):

        chunks_text += (
            f"\n\n"
            f"--- CHUNK ID: {index} ---\n"
            f"{chunk.page_content}\n"
        )

    # --------------------------------------------------------
    # Reranker prompt
    # --------------------------------------------------------

    prompt = f"""You are a document re-ranker.

You are given a user question and a list of
candidate knowledge-base chunks.

Your task is to rank ALL chunks according to
their relevance to the question.

The most directly useful chunk must come first.

Question:
{question}

Candidate chunks:
{chunks_text}

Return ONLY a JSON array of ALL chunk IDs from
most relevant to least relevant.

Do not omit any chunk.

Example:
[3, 1, 5, 2, 4]"""

    # --------------------------------------------------------
    # Manual structured output parsing
    # --------------------------------------------------------

    try:
        response = get_reranker_llm().invoke(
            [
                HumanMessage(content=prompt)
            ]
        )

        text = response.content
        match = re.search(r'\[\s*\d+(?:\s*,\s*\d+)*\s*\]', text)
        if match:
            order = json.loads(match.group())
        else:
            print("[rerank] Could not parse ranking, using default order.")
            order = list(range(1, len(chunks) + 1))
    except Exception as e:
        print(f"[rerank] Error during reranking: {e}")
        order = list(range(1, len(chunks) + 1))

    # --------------------------------------------------------
    # Convert IDs back to Documents
    # --------------------------------------------------------

    reranked_chunks = []

    for index in order:

        if 1 <= index <= len(chunks):

            reranked_chunks.append(
                chunks[index - 1]
            )

    return reranked_chunks


# ============================================================
# Optimized Retrieval Pipeline
# ============================================================

def fetch_context(
    original_question: str,
    history: list[dict] | None = None,
) -> list[Document]:
    """
    Optimized retrieval pipeline:

    1. Rewrite query
    2. Retrieve using original question
    3. Retrieve using rewritten question
    4. Merge results
    5. Rerank results
    6. Return top FINAL_K chunks
    """

    # --------------------------------------------------------
    # 1. Query Rewriting
    # --------------------------------------------------------

    rewritten_question = rewrite_query(
        original_question,
        history,
    )

    print("\n" + "=" * 70)
    print("ORIGINAL QUESTION:")
    print(original_question)

    print("\nREWRITTEN QUESTION:")
    print(rewritten_question)

    # --------------------------------------------------------
    # 2. Retrieve original question
    # --------------------------------------------------------

    chunks_original = fetch_context_unranked(
        original_question
    )

    # --------------------------------------------------------
    # 3. Retrieve rewritten question
    # --------------------------------------------------------

    chunks_rewritten = fetch_context_unranked(
        rewritten_question
    )

    # --------------------------------------------------------
    # 4. Merge results
    # --------------------------------------------------------

    chunks = merge_chunks(
        chunks_original,
        chunks_rewritten,
    )

    print(
        f"\nTotal candidate chunks after merge: "
        f"{len(chunks)}"
    )

    # --------------------------------------------------------
    # 5. Rerank
    # --------------------------------------------------------

    reranked_chunks = rerank(
        original_question,
        chunks,
    )

    # --------------------------------------------------------
    # 6. Return final chunks
    # --------------------------------------------------------

    final_chunks = reranked_chunks[:FINAL_K]

    print(
        f"Final chunks returned: "
        f"{len(final_chunks)}"
    )

    return final_chunks