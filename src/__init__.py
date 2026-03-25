from src.llm_factory import LLMEnum, LLMFactory
from src.parser_PDF import ParserPDF


def main():
    llm = LLMFactory.get_llm(LLMEnum.LLAMA_GROK)


if __name__ == "__main__":
    main()
