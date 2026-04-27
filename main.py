import argparse

from dotenv import load_dotenv
from langchain_core.language_models.chat_models import BaseChatModel
from src.evaluate import calculate_all_keywords_metrics, calculate_rouge_scores
from src.helpers import download_and_save_arxiv_pdf, download_arxiv_abstract, read, save
from src.key_words_finder import KeyWordsFinder
from src.llm_factory import LLMEnum, LLMFactory
from src.parser_PDF import ParserPDF
from src.summarizer.summarizer_factory import SummarizerEnum, SummarizerFactory
from src.text_object_type import TextObjectType


def download_document_and_abstract(arxiv_id: str, name: str):
    download_and_save_arxiv_pdf(arxiv_id, name)
    original_abstract = download_arxiv_abstract(arxiv_id)
    save(name, TextObjectType.ORIGINAL_ABSTRACT, original_abstract)


def clean_pdf(name: str, llm: BaseChatModel, chunk_size: int = 4000):
    parser = ParserPDF(llm, chunk_size)
    text = parser.read_markdown_from_file(name)
    cleaned = parser.clean_text(text)
    save(name, TextObjectType.CLEANED, cleaned)


def summarize_text(
    name: str, type: SummarizerEnum, llm: BaseChatModel, chunk_size: int = 4000
):
    summarizer = SummarizerFactory.get_summarizer(type, llm, chunk_size)
    text = read(name, TextObjectType.CLEANED)
    abstract = summarizer.build_abstract(text)
    save(
        name,
        TextObjectType.PRODUCED_ABSTRACT_ITERATIVE
        if type == SummarizerEnum.ITERATIVE
        else TextObjectType.PRODUCED_ABSTRACT_MAP_REDUCE,
        abstract,
    )


def find_keywords(name: str, llm: BaseChatModel, abstract_type: TextObjectType):
    finder = KeyWordsFinder(llm)
    produced_abstract = read(name, abstract_type)
    finder.find_and_save_keywords(name, produced_abstract)


def calc_metrics(name: str, abstract_type: TextObjectType):
    original_abstract = read(name, TextObjectType.ORIGINAL_ABSTRACT)
    produced_abstract = read(name, abstract_type)
    calculate_rouge_scores(name, original_abstract, produced_abstract)

    reference_keywords = read(name, TextObjectType.KEY_WORDS_REFERENCE)
    stat_keywords = read(name, TextObjectType.KEY_WORDS_STATISTICAL)
    llm_keywords = read(name, TextObjectType.KEY_WORDS_LLM)
    calculate_all_keywords_metrics(
        name, reference_keywords, stat_keywords, llm_keywords
    )


def main():
    load_dotenv()

    parser = argparse.ArgumentParser(
        description="NLP Pipeline: Download, Clean, Summarize and Evaluate ArXiv Papers"
    )

    parser.add_argument(
        "--name", default="test/test_document", help="Base name/path for files"
    )
    parser.add_argument(
        "--llm",
        default="GEMINI_25_FLASH",
        choices=[e.name for e in LLMEnum],
        help="LLM model to use",
    )

    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # 1. Download
    p_down = subparsers.add_parser(
        "download", help="Download PDF and original abstract"
    )
    p_down.add_argument("arxiv_id", help="ArXiv ID (e.g. 1706.03762)")

    # 2. Clean
    p_clean = subparsers.add_parser("clean", help="Clean PDF and convert to markdown")
    p_clean.add_argument(
        "--chunk", type=int, default=4000, help="Chunk size for processing"
    )

    # 3. Summarize
    p_sum = subparsers.add_parser("summarize", help="Create abstract from cleaned text")
    p_sum.add_argument(
        "--method",
        choices=[e.name for e in SummarizerEnum],
        default="ITERATIVE",
        help="Summarization method",
    )
    p_sum.add_argument("--chunk", type=int, default=4000, help="Chunk size")

    # 4. Keywords
    p_key = subparsers.add_parser(
        "keywords", help="Extract keywords from produced abstract"
    )
    p_key.add_argument(
        "--abs_type",
        choices=["ITERATIVE", "MAP_REDUCE"],
        default="ITERATIVE",
        help="Which abstract to use",
    )

    # 5. Metrics
    p_met = subparsers.add_parser("metrics", help="Calculate ROUGE and Keyword metrics")
    p_met.add_argument(
        "--abs_type",
        choices=["ITERATIVE", "MAP_REDUCE"],
        default="ITERATIVE",
        help="Which abstract to evaluate",
    )

    args = parser.parse_args()

    llm = None
    if args.command in ["clean", "summarize", "keywords"]:
        llm = LLMFactory.get_llm(LLMEnum[args.llm])

    if args.command == "download":
        download_document_and_abstract(args.arxiv_id, args.name)

    elif args.command == "clean":
        clean_pdf(args.name, llm, args.chunk)

    elif args.command == "summarize":
        method = SummarizerEnum[args.method]
        summarize_text(args.name, method, llm, args.chunk)

    elif args.command == "keywords":
        abs_type = (
            TextObjectType.PRODUCED_ABSTRACT_ITERATIVE
            if args.abs_type == "ITERATIVE"
            else TextObjectType.PRODUCED_ABSTRACT_MAP_REDUCE
        )
        find_keywords(args.name, llm, abs_type)

    elif args.command == "metrics":
        abs_type = (
            TextObjectType.PRODUCED_ABSTRACT_ITERATIVE
            if args.abs_type == "ITERATIVE"
            else TextObjectType.PRODUCED_ABSTRACT_MAP_REDUCE
        )
        calc_metrics(args.name, abs_type)

    else:
        parser.print_help()


if __name__ == "__main__":
    main()
