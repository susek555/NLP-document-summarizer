from rouge_score import rouge_scorer


def calculate_rouge_scores(name: str, original_abstract: str, produced_abstract: str):
    # rouge1: unigrams
    # rouge2: bigrams
    # rougeL: longest common subsequence

    scorer = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)
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


def calculate_all_keywords_metrics(
    name: str, reference_string: str, stat_string: str, llm_string: str
):
    scorer = rouge_scorer.RougeScorer(["rouge1", "rougeL"], use_stemmer=True)

    def process_input(raw_text):
        if not raw_text:
            return []
        return [k.strip() for k in raw_text.splitlines() if k.strip()]

    ref_list = process_input(reference_string)
    stat_list = process_input(stat_string)
    llm_list = process_input(llm_string)

    def get_metrics(r_list, p_list):
        ref_set = set(k.lower() for k in r_list)
        prod_set = set(k.lower() for k in p_list)

        hits = ref_set.intersection(prod_set)

        precision = len(hits) / len(prod_set) if prod_set else 0
        recall = len(hits) / len(ref_set) if ref_set else 0
        f1 = (
            2 * (precision * recall) / (precision + recall)
            if (precision + recall) > 0
            else 0
        )

        scores = scorer.score(" ".join(r_list), " ".join(p_list))

        return {
            "hits": list(hits),
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "rouge1": scores["rouge1"].fmeasure,
            "rougeL": scores["rougeL"].fmeasure,
        }

    stat_results = get_metrics(ref_list, stat_list)
    llm_results = get_metrics(ref_list, llm_list)

    report = f"# Keywords Comparison Report: {name.split('/')[-1]}\n\n"
    report += f"**Reference List:** `{', '.join(ref_list)}`  \n"
    report += f"**Stat (RAKE) List:** `{', '.join(stat_list)}`  \n"
    report += f"**LLM List:** `{', '.join(llm_list)}`  \n\n"

    report += "| Metrics | Statistical (RAKE) | LLM |\n"
    report += "| :--- | :--- | :--- |\n"
    report += f"| **Exact Matches** | {', '.join(stat_results['hits']) if stat_results['hits'] else 'None'} | {', '.join(llm_results['hits']) if llm_results['hits'] else 'None'} |\n"
    report += f"| **Precision** | {stat_results['precision']:.4f} | {llm_results['precision']:.4f} |\n"
    report += (
        f"| **Recall** | {stat_results['recall']:.4f} | {llm_results['recall']:.4f} |\n"
    )
    report += f"| **F1-Score** | {stat_results['f1']:.4f} | {llm_results['f1']:.4f} |\n"
    report += f"| **ROUGE-1** | {stat_results['rouge1']:.4f} | {llm_results['rouge1']:.4f} |\n"
    report += f"| **ROUGE-L** | {stat_results['rougeL']:.4f} | {llm_results['rougeL']:.4f} |\n"

    with open(f"{name}_keywords_comparison_report.md", "w", encoding="utf-8") as f:
        f.write(report)
