from src.helpers import get_text_content
from src.summarizer.summarizer import Summarizer


class MapReduceSummarizer(Summarizer):
    def _map(self, chunk: str) -> str:
        SYSTEM_PROMPT = (
            "Jesteś ekspertem od budowania streszczeń dokumentów "
            "metodą wzorowaną na map-reduce."
            "Teraz wykonujesz częsci fazy map."
            "W treści prompta otrzymasz fragment tekstu,"
            "który jest częścią całości dokumentu"
            "Twoim zadaniem jest zbudowanie streszczenia cząstkowego,"
            "które w następnym etapie zostanie połączone z innymi."
            "Odpowiedź ma zawierać zwięzłe streszczenie tego fragmentu."
            "ZASADY:\n"
            "1. Nie zmieniaj merytoryki tekstu.\n"
            "2. Nie zatracaj ważnych informacji zawartych w tekście.\n"
            "3. Bądź konserwatywny jeśli chodzi o długość streszczenia, "
            "czyli nie przekraczaj 2-3 zdań.\n"
            "4. Zwróć tylko czysty tekst w formacie Markdown.\n"
            "5. Zachowaj język oryginału, czyli jeśli tekst był po angielsku, "
            "dalej ma być po angielsku\n"
            "6. Zwróc tylko surowy tekst, nie dopisuj NIC od siebie.\n"
            "7. Twoja odpowiedź musi zawierać WYŁĄCZNIE wyprodukowane streszczenie. "
            "Nie dodawaj żadnych wyjaśnień, "
            "komentarzy ani metadanych o procesie czyszczenia.\n"
        )

        response = self.llm.invoke([("system", SYSTEM_PROMPT), ("human", chunk)])
        return get_text_content(response.content)

    def _reduce(self, partial_abstracts: list[str]):
        SYSTEM_PROMPT = (
            "Jesteś ekspertem od budowania streszczeń dokumentów "
            "metodą wzorowaną na map-reduce."
            "Teraz wykonujesz fazę reduce."
            "W treści prompta otrzymasz streszczenia cząstkowe,"
            "które zawierają najważniejsze informacje z fragmentów dokumentu."
            "Twoim zadaniem jest zbudowanie streszczenia końcowego,"
            "które połączy streszczenia cząstkowe i "
            "jak najlepiej odda treść całości dokumentu."
            "Odpowiedź ma zawierać TYLKO streszczenie końcowe."
            "ZASADY:\n"
            "1. Nie zmieniaj merytoryki tekstu.\n"
            "2. Nie zatracaj ważnych informacji zawartych w tekście.\n"
            "3. Bądź konserwatywny jeśli chodzi o długość streszczenia, "
            "czyli nie przekraczaj 10 zdań.\n"
            "4. Zwróć tylko czysty tekst w formacie Markdown.\n"
            "5. Zachowaj język oryginału, czyli jeśli tekst był po angielsku, "
            "dalej ma być po angielsku\n"
            "6. Zwróc tylko surowy tekst, nie dopisuj NIC od siebie.\n"
            "7. Twoja odpowiedź musi zawierać WYŁĄCZNIE wyprodukowane streszczenie. "
            "Nie dodawaj żadnych wyjaśnień, "
            "komentarzy ani metadanych o procesie czyszczenia.\n"
        )

        response = self.llm.invoke(
            [("system", SYSTEM_PROMPT), ("human", "\n".join(partial_abstracts))]
        )
        return get_text_content(response.content)

    def build_abstract(self, text: str) -> str:
        chunks = self.splitter.split_text(text)
        partial_abstracts = []

        print(f"Building abstract for {len(chunks)} chunks...")

        for i, chunk in enumerate(chunks):
            print(f"Summarizing chunk {i + 1}/{len(chunks)}...")
            partial_abstracts.append(self._map(chunk))

        print("Building final abstract...")
        return self._reduce(partial_abstracts)
