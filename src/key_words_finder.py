from langchain_core.language_models.chat_models import BaseChatModel
from multi_rake import Rake

from src.helpers import get_text_content
from src.text_object_type import TextObjectType


class KeyWordsFinder:
    def __init__(self, llm: BaseChatModel):
        self.llm = llm

    def _get_stat_keywords(self, text: str, language_code="en"):
        rake = Rake(language_code=language_code)
        keywords = rake.apply(text)
        return [kw[0] for kw in keywords[:5]]

    def _get_llm_keywords(self, text) -> str:
        SYSTEM_PROMPT = (
            "Otrzymasz tekst streszczenia."
            "Twoim zadaniem jest wyznaczyć 5 słów kluczowych"
            "najbardziej wartościowych dla tekstu."
            "Nie dodawaj żadnego wprowadzenia ani nic od siebie,"
            "odpowiedź ma zawierać TYLKO I WYŁĄCZNIE 5 wybranych słów kluczowych"
            "w oryginalnym języku streszczenia."
        )

        response = self.llm.invoke([("system", SYSTEM_PROMPT), ("human", text)])
        return get_text_content(response.content)

    def find_and_save_keywords(self, name: str, text: str):
        stat = self._get_stat_keywords(text)
        llm = self._get_llm_keywords(text)

        with open(f"{name}{TextObjectType.KEY_WORDS_STATISTICAL.value}", "w") as f:
            f.write("\n".join(stat))

        with open(f"{name}{TextObjectType.KEY_WORDS_LLM.value}", "w") as f:
            f.write(llm)
