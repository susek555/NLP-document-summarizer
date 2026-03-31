from dotenv import load_dotenv
from src.llm_factory import LLMEnum, LLMFactory
from src.parser_PDF import ParserPDF
from src.summarizer.iterative_summarizer import IterativeSummarizer


def clean_pdf():
    load_dotenv()
    # llm = LLMFactory.get_llm(LLMEnum.LLAMA_GROK)
    llm = LLMFactory.get_llm(LLMEnum.GEMINI_25_FLASH)
    parser = ParserPDF(llm)
    text = parser.read_markdown_from_file("test/test_document.pdf")
    cleaned = parser.clean_text(text)
    print(cleaned)
    with open("test/result_v1.md", "w") as f:
        f.write(cleaned)

def summarize_text():
    load_dotenv()
    llm = LLMFactory.get_llm(LLMEnum.GEMINI_25_FLASH)
    summarizer = IterativeSummarizer(llm)
    text = ""
    with open("test/result_v1.md") as f:
        text = f.read()
    abstract = summarizer.build_abstract(text)
    print(abstract)
    with open("test/iterative_summarizer_result_1.md", "w") as f:
        f.write(abstract)

if __name__ == "__main__":
    # clean_pdf()
    summarize_text()
