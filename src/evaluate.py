from rouge_score import rouge_scorer


def calculate_rouge_scores(name: str, original_abstract: str, produced_abstract: str):
    # rouge1: unigrams
    # rouge2: bigrams
    # rougeL: longest common subsequence

    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
    scores = scorer.score(original_abstract, produced_abstract)

    result = ""
    for measure, metrics in scores.items():
        result += f"--- {measure} ---\n"
        result += f"Precision: {metrics.precision:.4f}\n"
        result += f"Recall:    {metrics.recall:.4f}\n"
        result += f"F1-Score:  {metrics.fmeasure:.4f}\n"
        result += "\n"

    with open(f"{name}_rouge_metrics.md", "w") as f:
        f.write(result)



