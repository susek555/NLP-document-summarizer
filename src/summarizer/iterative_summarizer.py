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
            "4. Bądź konserwatywny jeśli chodzi o długość streszczenia, "
            "czyli w każdej iteracji możesz dopisać maksymalnie 2 zdania.\n"
            "5. Zwróć tylko czysty tekst w formacie Markdown.\n"
            "6. Zachowaj język oryginału, czyli jeśli tekst był po angielsku, "
            "dalej ma być po angielsku\n"
            "7. Zwróc tylko surowy tekst, nie dopisuj NIC od siebie.\n"
            "8. Twoja odpowiedź musi zawierać WYŁĄCZNIE przefiltrowany tekst. "
            "Jeśli po filtracji nie pozostanie żaden tekst, "
            "Twoja odpowiedź musi być całkowicie pusta (zero znaków). "
            "Nie dodawaj żadnych wyjaśnień, "
            "komentarzy ani metadanych o procesie czyszczenia.\n"
        )

        human_prompt = (
            "Abstract: \n" + abstract + "\n \n \n" + "New text chunk: \n" + chunk
        )

        response = self.llm.invoke([("system", SYSTEM_PROMPT), ("human", human_prompt)])
        return response.content

    def build_abstract(self, text: str) -> str:
        chunks = self.splitter.split_text(text)
        abstract = ""

        print(f"Building abstract for {len(chunks)} chunks...")

        for i, chunk in enumerate(chunks):
            print(f"Summarizing chunk {i + 1}/{len(chunks)}...")
            abstract = self._advance_abstract(abstract, chunk)

        return abstract
