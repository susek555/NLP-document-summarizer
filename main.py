from dotenv import load_dotenv
from src.llm_factory import LLMEnum, LLMFactory
from src.parser_PDF import ParserPDF


def main():
    load_dotenv()
    llm = LLMFactory.get_llm(LLMEnum.LLAMA_GROK)


if __name__ == "__main__":
    main()
