from dotenv import load_dotenv
from src.llm_factory import LLMEnum, LLMFactory
from src.parser_PDF import ParserPDF


def main():
    load_dotenv()
    llm = LLMFactory.get_llm(LLMEnum.LLAMA_GROK)
    parser = ParserPDF(llm)
    text = parser.read_markdown_from_file("test/test_document.pdf")
    print(parser._clean_chunk(text[:4000]))


if __name__ == "__main__":
    main()
