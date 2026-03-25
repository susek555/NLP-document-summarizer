import pymupdf4llm
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_text_splitters import RecursiveCharacterTextSplitter


class ParserPDF:
    def __init__(self, llm: BaseChatModel):
        self.llm = llm
        self.splitter = RecursiveCharacterTextSplitter(
            separators=["\n#", "\n\n", "\n", " ", ""]
        )

    def read_markdown_from_file(self, filepath: str) -> str:
        return pymupdf4llm.to_markdown(filepath)

    def _clean_chunk(self, chunk: str) -> str:
        SYSTEM_PROMPT = (
            "Jesteś ekspertem od czyszczenia danych tekstowych. "
            "Twoim zadaniem jest usunięcie "
            "szumu z tekstu wyciągniętego z PDF. Usuń: numery stron, stopki, nagłówki, "
            "identyfikatory bibliograficzne pojawiające się co stronę. \n"
            "ZASADY:\n"
            "1. Nie zmieniaj merytoryki tekstu.\n"
            "2. Nie streszczaj!\n"
            "3. Napraw rozbite wyrazy (np. 'stresz- czenie' -> 'streszczenie').\n"
            "4. Zwróć tylko czysty tekst w formacie Markdown."
        )

        response = self.llm.invoke([("system", SYSTEM_PROMPT), ("human", chunk)])
        return response.content

    # def clean_text(self, text: str) -> str:


# if __name__ == "__main__":
#     parser = ParserPDF()
#     text = parser.read_markdown_from_file("test/test_document.pdf")
#     print(text[:10000])
