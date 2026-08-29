# 🏢 InsureLLM Expert Assistant

A fully local **Retrieval-Augmented Generation (RAG)** knowledge assistant for **Insurellm**, an insurance technology company. The system answers questions about employees, projects, contracts, and performance using only the company's internal knowledge base — powered entirely by **Ollama**, with no external API calls required.
---

## ✨ Features

- **Local-first RAG pipeline** — no OpenAI/Anthropic API keys needed, everything runs through [Ollama](https://ollama.com/).
- **Query rewriting** — the user's question is rewritten into a precise, self-contained search query using conversation history.
- **Dual retrieval** — the system retrieves using both the original and the rewritten query, then merges and de-duplicates the results.
- **LLM-based reranking** — retrieved chunks are re-ordered by an LLM judge for relevance before being passed to the generator.
- **Conversational memory** — full chat history is passed to the generation model for context-aware answers.
- **Polished dark-mode chat UI** — built with Gradio, showing live model info, retrieved sources, and quick suggestion prompts.
- **Built-in evaluation suite** — automated retrieval metrics (MRR, nDCG, keyword coverage) and LLM-judged answer quality (accuracy, completeness, relevance), visualized in a separate Gradio dashboard.

---

## 🏗️ Architecture

```
┌─────────────┐      ┌──────────────────┐      ┌─────────────────┐
│  Markdown   │─────▶│     ingest.py     │─────▶│  Chroma Vector   │
│  Knowledge  │      │ (chunk + embed)   │      │      Store       │
│    Base     │      └──────────────────┘      └────────┬─────────┘
└─────────────┘                                          │
                                                          ▼
┌─────────────┐      ┌──────────────────┐      ┌─────────────────┐
│    User     │─────▶│   retrieval.py     │◀────│   Query Rewrite  │
│  Question   │      │ retrieve → merge   │      │   (qwen3:4b)     │
└─────────────┘      │   → rerank         │      └─────────────────┘
                      └────────┬───────────┘
                               ▼
                      ┌──────────────────┐
                      │    answer.py      │
                      │ (RAG generation)  │
                      └────────┬───────────┘
                               ▼
                      ┌──────────────────┐
                      │  app.py (Gradio)  │
                      │   Chat Interface  │
                      └──────────────────┘
```

### Retrieval pipeline (`retrieval.py`)

1. **Rewrite** the user's question into a precise, standalone search query (uses conversation history).
2. **Retrieve** top candidates using both the original and rewritten queries.
3. **Merge** results and remove duplicates.
4. **Rerank** all candidates with an LLM judge.
5. Return the top `FINAL_K` chunks to the generator.

---

## 🧰 Tech Stack

| Layer | Technology |
|---|---|
| LLM runtime | [Ollama](https://ollama.com/) (local) |
| Generation & Reranking model | `qwen3:4b` |
| Embedding model | `qwen3-embedding:0.6b` |
| Vector store | [Chroma](https://www.trychroma.com/) via `langchain-chroma` |
| Orchestration | [LangChain](https://www.langchain.com/) |
| UI | [Gradio](https://www.gradio.app/) |
| Package management | [uv](https://docs.astral.sh/uv/) |

---

## 📁 Project Structure

```
InsureLLM-RAG-Knowledge-Assistant/
│
├── app.py                   # Main Gradio chat interface
├── ingest.py                 # Loads, chunks, embeds & stores the knowledge base
├── retrieval.py               # Query rewriting + retrieval + reranking pipeline
├── answer.py                  # RAG answer generation logic
│
├── evaluation/
│   ├── __init__.py
│   ├── test.py                # Test-case schema & loader (tests.jsonl)
│   ├── eval.py                 # Retrieval & answer evaluation logic (MRR, nDCG, LLM judge)
│   ├── eval_app.py              # Gradio dashboard for evaluation results
│   └── tests.jsonl              # Evaluation test set
│
├── knowledge-base/              # Source markdown documents (company data)
│
├── preprocessed_db/              # Chroma vector database (generated — not committed)
│
├── .env.example                  # Example environment variables
├── .gitignore
├── pyproject.toml                 # Project dependencies (uv)
└── README.md
```

---

## 🚀 Getting Started

### Prerequisites

- Python 3.11+
- [uv](https://docs.astral.sh/uv/getting-started/installation/) installed
- [Ollama](https://ollama.com/download) installed and running

### 1. Clone the repository

```bash
git clone https://github.com/<your-username>/InsureLLM-RAG-Knowledge-Assistant.git
cd InsureLLM-RAG-Knowledge-Assistant
```

### 2. Install dependencies

```bash
uv sync
```

### 3. Pull the required Ollama models

```bash
ollama pull qwen3:4b
ollama pull qwen3-embedding:0.6b
```

### 4. Set up environment variables

```bash
cp .env.example .env
```

### 5. Build the knowledge base (vector database)

Place your `.md` knowledge files inside `knowledge-base/`, then run:

```bash
uv run ingest.py
```

This creates the Chroma vector store in `preprocessed_db/`.

### 6. Launch the chat assistant

```bash
uv run app.py
```

The app will open automatically at `http://127.0.0.1:7860`.

---

## 📊 Running Evaluations

The project includes an automated evaluation suite comparing retrieval quality and answer quality against a labeled test set (`evaluation/tests.jsonl`).

### Via the evaluation dashboard

```bash
uv run evaluation/eval_app.py
```

This opens a Gradio dashboard where you can run:
- **Retrieval Evaluation** — MRR, nDCG, and keyword coverage, broken down by category.
- **Answer Evaluation** — LLM-judged accuracy, completeness, and relevance, broken down by category.

### Via the CLI (single test case)

```bash
uv run evaluation/eval.py <test_number>
```

---

## ⚙️ Configuration

Key parameters can be adjusted at the top of each file:

| Variable | File | Description |
|---|---|---|
| `RETRIEVAL_K` | `retrieval.py` | Number of candidate chunks retrieved per query before reranking |
| `FINAL_K` | `retrieval.py` | Number of chunks kept after reranking |
| `CHUNK_SIZE` / `CHUNK_OVERLAP` | `ingest.py` | Text splitting parameters |
| `MODEL` | `answer.py`, `retrieval.py` | Generation / reranking model name |
| `EMBEDDING_MODEL` | `ingest.py`, `retrieval.py` | Embedding model name |

---

## 📄 License

This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.

---

## 🙌 Acknowledgements

Built as part of an LLM Engineering learning project, exploring local RAG pipelines with query rewriting, dual retrieval, and LLM-based reranking.
