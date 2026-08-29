import gradio as gr
import pandas as pd
from collections import defaultdict

from evaluation.eval import (
    evaluate_all_retrieval,
    evaluate_all_answers,
)


# ============================================================
# Retrieval Evaluation
# ============================================================

def run_retrieval_evaluation(
    progress=gr.Progress()
):

    total_mrr = 0
    total_ndcg = 0
    total_coverage = 0

    category_scores = defaultdict(list)

    count = 0

    for (
        test,
        result,
        prog
    ) in evaluate_all_retrieval():

        count += 1

        total_mrr += result.mrr
        total_ndcg += result.ndcg
        total_coverage += result.keyword_coverage

        category_scores[
            test.category
        ].append(result.mrr)

        progress(
            prog,
            desc=f"Evaluating test {count}..."
        )

    if count == 0:

        return (
            "No evaluation tests found.",
            pd.DataFrame()
        )

    avg_mrr = total_mrr / count
    avg_ndcg = total_ndcg / count
    avg_coverage = total_coverage / count

    metrics = f"""
    ## Retrieval Results

    **Tests:** {count}

    ### MRR
    `{avg_mrr:.4f}`

    ### nDCG
    `{avg_ndcg:.4f}`

    ### Keyword Coverage
    `{avg_coverage:.1f}%`
    """

    chart_data = []

    for category, scores in category_scores.items():

        chart_data.append(
            {
                "Category": category,
                "Average MRR": sum(scores) / len(scores),
            }
        )

    df = pd.DataFrame(
        chart_data
    )

    return metrics, df


# ============================================================
# Answer Evaluation
# ============================================================

def run_answer_evaluation(
    progress=gr.Progress()
):

    total_accuracy = 0
    total_completeness = 0
    total_relevance = 0

    category_scores = defaultdict(list)

    count = 0

    for (
        test,
        result,
        prog
    ) in evaluate_all_answers():

        count += 1

        total_accuracy += result.accuracy
        total_completeness += result.completeness
        total_relevance += result.relevance

        category_scores[
            test.category
        ].append(result.accuracy)

        progress(
            prog,
            desc=f"Evaluating test {count}..."
        )

    if count == 0:

        return (
            "No evaluation tests found.",
            pd.DataFrame()
        )

    avg_accuracy = (
        total_accuracy / count
    )

    avg_completeness = (
        total_completeness / count
    )

    avg_relevance = (
        total_relevance / count
    )

    metrics = f"""
    ## Answer Results

    **Tests:** {count}

    ### Accuracy
    `{avg_accuracy:.2f}/5`

    ### Completeness
    `{avg_completeness:.2f}/5`

    ### Relevance
    `{avg_relevance:.2f}/5`
    """

    chart_data = []

    for category, scores in category_scores.items():

        chart_data.append(
            {
                "Category": category,
                "Average Accuracy":
                    sum(scores) / len(scores),
            }
        )

    df = pd.DataFrame(
        chart_data
    )

    return metrics, df


# ============================================================
# Gradio App
# ============================================================

def main():

    with gr.Blocks(
        title="InsureLLM RAG Evaluation"
    ) as app:

        gr.Markdown(
            "# 📊 InsureLLM RAG Evaluation"
        )

        gr.Markdown(
            "Evaluate retrieval and answer quality "
            "of the InsureLLM RAG system."
        )

        # ----------------------------------------------------
        # Retrieval
        # ----------------------------------------------------

        gr.Markdown(
            "## 🔍 Retrieval Evaluation"
        )

        retrieval_button = gr.Button(
            "Run Retrieval Evaluation",
            variant="primary"
        )

        retrieval_metrics = gr.Markdown()

        retrieval_chart = gr.BarPlot(
            x="Category",
            y="Average MRR",
            title="Average MRR by Category",
            y_lim=[0, 1],
        )

        retrieval_button.click(
            fn=run_retrieval_evaluation,
            outputs=[
                retrieval_metrics,
                retrieval_chart,
            ],
        )

        # ----------------------------------------------------
        # Answer
        # ----------------------------------------------------

        gr.Markdown(
            "## 💬 Answer Evaluation"
        )

        answer_button = gr.Button(
            "Run Answer Evaluation",
            variant="primary"
        )

        answer_metrics = gr.Markdown()

        answer_chart = gr.BarPlot(
            x="Category",
            y="Average Accuracy",
            title="Average Accuracy by Category",
            y_lim=[0, 5],
        )

        answer_button.click(
            fn=run_answer_evaluation,
            outputs=[
                answer_metrics,
                answer_chart,
            ],
        )

    app.launch(
        inbrowser=True
    )


if __name__ == "__main__":

    main()