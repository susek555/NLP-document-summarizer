from enum import Enum


class TextObjectType(Enum):
    BASE_PDF = ".pdf",
    CLEANED_MD = "_cleaned.md"
    ORIGINAL_ABSTRACT_MD = "_original_abstract.md"
    PRODUCED_ABSTRACT_MD = "_produced_abstract.md"
    KEY_WORDS_STATISTICAL = "_key_words_statistical.md"
    KEY_WORDS_LLM = "_key_words_llm.md"
