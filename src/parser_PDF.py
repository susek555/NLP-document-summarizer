import pymupdf4llm
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_text_splitters import RecursiveCharacterTextSplitter

from src.text_object_type import TextObjectType


class ParserPDF:
    def __init__(self, llm: BaseChatModel, chunk_size: int = 4000):
        self.llm = llm
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=400,
            separators=["\n#", "\n\n", "\n", " ", ""],
        )

    @staticmethod
    def read_markdown_from_file(name: str) -> str:
        return pymupdf4llm.to_markdown(f"{name}{TextObjectType.BASE_PDF.value}")

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
            "4. Zwróć tylko czysty tekst w formacie Markdown.\n"
            "5. Zachowaj język oryginału, czyli jeśli tekst był po angielsku, "
            "dalej ma być po angielsku\n"
            "6. Usuń spis literatury, wykaz źródeł.\n"
            "7. Usuń adnotacje zawarte w tekście (np. numery w środku tekstu).\n"
            "8. Zwróc tylko surowy tekst, nie dopisuj NIC od siebie.\n"
            "9. Usuń fragmenty, które nie mają "
            "przyczynowo-skutkowego powiązania z resztą tekstu.\n"
            "10. Twoja odpowiedź musi zawierać WYŁĄCZNIE przefiltrowany tekst. "
            "Jeśli po filtracji nie pozostanie żaden tekst, "
            "Twoja odpowiedź musi być całkowicie pusta (zero znaków). "
            "Nie dodawaj żadnych wyjaśnień, "
            "komentarzy ani metadanych o procesie czyszczenia.\n"
        )

        response = self.llm.invoke([("system", SYSTEM_PROMPT), ("human", chunk)])

        if isinstance(response.content, list):
            return "".join(
                [
                    part.get("text", "") if isinstance(part, dict) else str(part)
                    for part in response.content
                ]
            )
        return str(response.content)

    def clean_text(self, text: str) -> str:
        chunks = self.splitter.split_text(text)
        cleaned_chunks = []

        print(f"Cleaning {len(chunks)} chunks...")

        for i, chunk in enumerate(chunks):
            print(f"Cleaning chunk {i + 1}/{len(chunks)}...")
            cleaned = self._clean_chunk(chunk)
            if cleaned.strip():
                cleaned_chunks.append(cleaned)

        return "\n\n".join(cleaned_chunks)

