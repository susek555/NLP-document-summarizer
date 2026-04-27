from src.helpers import get_text_content
from src.summarizer.summarizer import Summarizer


class IterativeSummarizer(Summarizer):
    def _advance_abstract(self, abstract: str, chunk: str) -> str:
        SYSTEM_PROMPT = (
            "Jesteś ekspertem od iteracyjnego budowania streszczeń dokumentów. "
            "W treści prompta otrzymasz abstrakt dotychczas przetworzonej treści"
            "oraz nową porcję tekstu, który trzeba dołożyć do streszczenia."
            "Twoim zadaniem jest edycja abstraktu w taki sposób, żeby jego"
            "treść została wzbogacona o informacje z nowego kawałka tekstu."
            "ZASADY:\n"
            "1. Nie zmieniaj merytoryki tekstu.\n"
            "2. Nie zatracaj informacji zawartych w streszczeniu.\n"
            "3. Jeśli to pierwszy fragment tekstu,"
            "dotychczasowe streszczenie będzie puste.\n"
            "4. LIMIT DŁUGOŚCI: Całe streszczenie musi "
            "liczyć od 5 do MAKSYMALNIE 10 ZDAŃ. "
            "Jeśli dodanie nowych informacji spowodowałoby "
            "przekroczenie 10 zdań, musisz przeredagować "
            "całość tak, aby zmieścić najważniejsze fakty w wyznaczonym limicie.\n"
            "5. Zwróć tylko czysty tekst w formacie Markdown.\n"
            "6. Zachowaj język oryginału, czyli jeśli tekst był po angielsku, "
            "dalej ma być po angielsku\n"
            "7. Zwróc tylko surowy tekst, nie dopisuj NIC od siebie.\n"
            "8. Twoja odpowiedź musi zawierać WYŁĄCZNIE opracowane streszczenie. "
            "Nie dodawaj żadnych wyjaśnień, "
            "komentarzy ani metadanych o procesie budowy streszczenia.\n"
        )

        human_prompt = (
            "Abstract: \n" + abstract + "\n \n \n" + "New text chunk: \n" + chunk
        )

        response = self.llm.invoke([("system", SYSTEM_PROMPT), ("human", human_prompt)])
        return get_text_content(response.content)

    def build_abstract(self, text: str) -> str:
        chunks = self.splitter.split_text(text)
        abstract = ""

        print(f"Building abstract for {len(chunks)} chunks...")

        for i, chunk in enumerate(chunks):
            print(f"Summarizing chunk {i + 1}/{len(chunks)}...")
            abstract = self._advance_abstract(abstract, chunk)

        return abstract
