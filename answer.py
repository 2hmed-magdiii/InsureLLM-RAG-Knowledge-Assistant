from langchain_ollama import ChatOllama
from langchain_core.documents import Document
from langchain_core.messages import (
    SystemMessage,
    HumanMessage,
    AIMessage,
)

from retrieval import fetch_context


# ============================================================
# Configuration
# ============================================================

MODEL = "qwen3:4b"

OLLAMA_BASE_URL = "http://localhost:11434"


# ============================================================
# System Prompt
# ============================================================

SYSTEM_PROMPT = """You are a knowledgeable and helpful assistant
representing Insurellm, an insurance technology company.

Your job is to answer questions using the provided
company knowledge base.

Follow these rules:

1. Use the provided context to answer the question.

2. Do not invent information that is not supported
   by the context.

3. If the answer cannot be found in the context,
   say that you don't know.

4. Give a clear and concise answer.

5. Do not rely on your general knowledge when the
   information is not available in the context.

Context:

{context}
"""


# ============================================================
# LLM
# ============================================================

llm = ChatOllama(
    model=MODEL,
    temperature=0,
    base_url=OLLAMA_BASE_URL,
)


# ============================================================
# Combine Conversation History
# ============================================================

def combined_question(
    question: str,
    history: list[dict] | None = None,
) -> str:
    """
    Combine previous user questions with the
    current question.

    This is kept for compatibility with the
    original implementation.
    """

    if not history:
        return question

    prior = "\n".join(
        message["content"]
        for message in history
        if message["role"] == "user"
    )

    return prior + "\n" + question


# ============================================================
# Answer Question
# ============================================================

def answer_question(
    question: str,
    history: list[dict] | None = None,
) -> tuple[str, list[Document]]:
    """
    Answer a question using the optimized RAG pipeline.

    Returns:
        answer
        retrieved documents
    """

    if history is None:
        history = []

    # --------------------------------------------------------
    # 1. Retrieve relevant documents
    # --------------------------------------------------------

    try:
        documents = fetch_context(
            question,
            history,
        )
    except Exception as e:
        print(f"[answer_question] Retrieval failed: {e}")
        return (
            "Sorry, I couldn't retrieve relevant documents. "
            "Please make sure the vector database is built and Ollama is running.",
            [],
        )

    # --------------------------------------------------------
    # 2. Build context
    # --------------------------------------------------------

    context = "\n\n".join(
        document.page_content
        for document in documents
    )

    # --------------------------------------------------------
    # 3. Build system prompt
    # --------------------------------------------------------

    system_prompt = SYSTEM_PROMPT.format(
        context=context
    )

    messages = [
        SystemMessage(
            content=system_prompt
        )
    ]

    # --------------------------------------------------------
    # 4. Add conversation history (both user and assistant)
    # --------------------------------------------------------

    for message in history:

        if message["role"] == "user":

            messages.append(
                HumanMessage(
                    content=message["content"]
                )
            )

        elif message["role"] == "assistant":

            messages.append(
                AIMessage(
                    content=message["content"]
                )
            )

    # --------------------------------------------------------
    # 5. Add current question
    # --------------------------------------------------------

    messages.append(
        HumanMessage(
            content=question
        )
    )

    # --------------------------------------------------------
    # 6. Generate answer
    # --------------------------------------------------------

    try:
        response = llm.invoke(
            messages
        )
        answer = response.content
    except Exception as e:
        print(f"[answer_question] Generation failed: {e}")
        answer = (
            "Sorry, I encountered an error while generating the answer. "
            "Please check that Ollama is running and the model is available."
        )

    return (
        answer,
        documents,
    )