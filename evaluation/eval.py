import math
import sys

from pydantic import BaseModel, Field
from langchain_ollama import ChatOllama

from evaluation.test import TestQuestion, load_tests
from retrieval import fetch_context
from answer import answer_question


# ============================================================
# Configuration
# ============================================================

JUDGE_MODEL = "qwen3:4b"

RETRIEVAL_K = 5


# ============================================================
# LLM Judge
# ============================================================

judge_llm = ChatOllama(
    model=JUDGE_MODEL,
    temperature=0,
    base_url="http://localhost:11434",
)


# ============================================================
# Evaluation Models
# ============================================================

class RetrievalEval(BaseModel):
    """
    Evaluation metrics for retrieval.
    """

    mrr: float = Field(
        description="Mean Reciprocal Rank"
    )

    ndcg: float = Field(
        description="Normalized Discounted Cumulative Gain"
    )

    keywords_found: int = Field(
        description="Number of keywords found"
    )

    total_keywords: int = Field(
        description="Total number of keywords"
    )

    keyword_coverage: float = Field(
        description="Percentage of keywords found"
    )


class AnswerEval(BaseModel):
    """
    Evaluation of generated answer.
    """

    feedback: str = Field(
        description="Feedback about the answer"
    )

    accuracy: float = Field(
        description="Factual correctness from 1 to 5"
    )

    completeness: float = Field(
        description="Completeness from 1 to 5"
    )

    relevance: float = Field(
        description="Relevance from 1 to 5"
    )


# ============================================================
# MRR
# ============================================================

def calculate_mrr(
    keyword: str,
    retrieved_docs: list
) -> float:

    keyword_lower = keyword.lower()

    for rank, doc in enumerate(
        retrieved_docs,
        start=1
    ):

        if keyword_lower in doc.page_content.lower():

            return 1.0 / rank

    return 0.0


# ============================================================
# DCG
# ============================================================

def calculate_dcg(
    relevances: list[int],
    k: int
) -> float:

    dcg = 0.0

    for i in range(
        min(k, len(relevances))
    ):

        dcg += (
            relevances[i]
            / math.log2(i + 2)
        )

    return dcg


# ============================================================
# nDCG
# ============================================================

def calculate_ndcg(
    keyword: str,
    retrieved_docs: list,
    k: int = RETRIEVAL_K
) -> float:

    keyword_lower = keyword.lower()

    relevances = [
        1
        if keyword_lower in doc.page_content.lower()
        else 0
        for doc in retrieved_docs[:k]
    ]

    dcg = calculate_dcg(
        relevances,
        k
    )

    ideal_relevances = sorted(
        relevances,
        reverse=True
    )

    idcg = calculate_dcg(
        ideal_relevances,
        k
    )

    if idcg == 0:

        return 0.0

    return dcg / idcg


# ============================================================
# Retrieval Evaluation
# ============================================================

def evaluate_retrieval(
    test: TestQuestion,
    k: int = RETRIEVAL_K
) -> RetrievalEval:

    retrieved_docs = fetch_context(
        test.question
    )

    # MRR
    mrr_scores = [
        calculate_mrr(
            keyword,
            retrieved_docs
        )
        for keyword in test.keywords
    ]

    avg_mrr = (
        sum(mrr_scores) / len(mrr_scores)
        if mrr_scores
        else 0.0
    )

    # nDCG
    ndcg_scores = [
        calculate_ndcg(
            keyword,
            retrieved_docs,
            k
        )
        for keyword in test.keywords
    ]

    avg_ndcg = (
        sum(ndcg_scores) / len(ndcg_scores)
        if ndcg_scores
        else 0.0
    )

    # Keyword coverage
    keywords_found = sum(
        1
        for score in mrr_scores
        if score > 0
    )

    total_keywords = len(
        test.keywords
    )

    keyword_coverage = (
        keywords_found / total_keywords * 100
        if total_keywords
        else 0.0
    )

    return RetrievalEval(
        mrr=avg_mrr,
        ndcg=avg_ndcg,
        keywords_found=keywords_found,
        total_keywords=total_keywords,
        keyword_coverage=keyword_coverage,
    )


# ============================================================
# Answer Evaluation
# ============================================================

def evaluate_answer(
    test: TestQuestion
) -> tuple[AnswerEval, str, list]:

    generated_answer, retrieved_docs = answer_question(
        test.question
    )

    judge_prompt = f"""
You are an expert evaluator for an enterprise
RAG question-answering system.

Evaluate the generated answer using the question,
reference answer, and retrieved context.

Question:
{test.question}

Generated Answer:
{generated_answer}

Reference Answer:
{test.reference_answer}

Retrieved Context:
{"".join(
    f"\n\n--- Document ---\n{doc.page_content}"
    for doc in retrieved_docs
)}

Evaluate the answer using these criteria.

Accuracy:
How factually correct is the answer compared
with the reference answer?

Completeness:
Does the answer contain all important information
needed to answer the question?

Relevance:
Does the answer directly answer the question
without unnecessary information?

Use a score from 1 to 5.

1 = Very poor
2 = Poor
3 = Acceptable
4 = Very good
5 = Excellent

Return the result using this exact format:

Accuracy: <score>
Completeness: <score>
Relevance: <score>

Feedback:
<short explanation>
"""

    response = judge_llm.invoke(
        judge_prompt
    )

    text = response.content

    # --------------------------------------------------------
    # Parse judge response
    # --------------------------------------------------------

    accuracy = extract_score(
        text,
        "Accuracy"
    )

    completeness = extract_score(
        text,
        "Completeness"
    )

    relevance = extract_score(
        text,
        "Relevance"
    )

    feedback = text

    result = AnswerEval(
        feedback=feedback,
        accuracy=accuracy,
        completeness=completeness,
        relevance=relevance,
    )

    return (
        result,
        generated_answer,
        retrieved_docs,
    )


# ============================================================
# Extract Score
# ============================================================

def extract_score(
    text: str,
    metric: str
) -> float:

    for line in text.splitlines():

        if line.lower().startswith(
            metric.lower()
        ):

            try:

                value = (
                    line.split(":")[1]
                    .strip()
                )

                return float(
                    value.split("/")[0]
                )

            except (
                ValueError,
                IndexError
            ):

                pass

    return 0.0


# ============================================================
# Evaluate All Retrieval Tests
# ============================================================

def evaluate_all_retrieval():

    tests = load_tests()

    total = len(tests)

    for index, test in enumerate(tests):

        result = evaluate_retrieval(
            test
        )

        progress = (
            (index + 1) / total
            if total
            else 1.0
        )

        yield (
            test,
            result,
            progress
        )


# ============================================================
# Evaluate All Answers
# ============================================================

def evaluate_all_answers():

    tests = load_tests()

    total = len(tests)

    for index, test in enumerate(tests):

        result = evaluate_answer(
            test
        )[0]

        progress = (
            (index + 1) / total
            if total
            else 1.0
        )

        yield (
            test,
            result,
            progress
        )


# ============================================================
# CLI Evaluation
# ============================================================

def run_cli_evaluation(
    test_number: int
):

    tests = load_tests()

    if (
        test_number < 0
        or test_number >= len(tests)
    ):

        print(
            f"Error: test number must be between "
            f"0 and {len(tests) - 1}"
        )

        sys.exit(1)

    test = tests[test_number]

    print("\n" + "=" * 80)
    print(f"Test #{test_number}")
    print("=" * 80)

    print(
        f"\nQuestion:\n{test.question}"
    )

    print(
        f"\nKeywords:\n{test.keywords}"
    )

    print(
        f"\nCategory:\n{test.category}"
    )

    print(
        f"\nReference Answer:\n"
        f"{test.reference_answer}"
    )

    # --------------------------------------------------------
    # Retrieval
    # --------------------------------------------------------

    print("\n" + "=" * 80)
    print("Retrieval Evaluation")
    print("=" * 80)

    retrieval_result = evaluate_retrieval(
        test
    )

    print(
        f"MRR: {retrieval_result.mrr:.4f}"
    )

    print(
        f"nDCG: {retrieval_result.ndcg:.4f}"
    )

    print(
        f"Keywords Found: "
        f"{retrieval_result.keywords_found}/"
        f"{retrieval_result.total_keywords}"
    )

    print(
        f"Keyword Coverage: "
        f"{retrieval_result.keyword_coverage:.1f}%"
    )

    # --------------------------------------------------------
    # Answer
    # --------------------------------------------------------

    print("\n" + "=" * 80)
    print("Answer Evaluation")
    print("=" * 80)

    (
        answer_result,
        generated_answer,
        retrieved_docs,
    ) = evaluate_answer(test)

    print(
        f"\nGenerated Answer:\n"
        f"{generated_answer}"
    )

    print(
        f"\nAccuracy: "
        f"{answer_result.accuracy:.2f}/5"
    )

    print(
        f"Completeness: "
        f"{answer_result.completeness:.2f}/5"
    )

    print(
        f"Relevance: "
        f"{answer_result.relevance:.2f}/5"
    )

    print(
        f"\nFeedback:\n"
        f"{answer_result.feedback}"
    )


# ============================================================
# Main
# ============================================================

def main():

    if len(sys.argv) != 2:

        print(
            "Usage: "
            "uv run evaluation/eval.py <test_number>"
        )

        sys.exit(1)

    try:

        test_number = int(
            sys.argv[1]
        )

    except ValueError:

        print(
            "Error: test number must be an integer"
        )

        sys.exit(1)

    run_cli_evaluation(
        test_number
    )


if __name__ == "__main__":

    main()