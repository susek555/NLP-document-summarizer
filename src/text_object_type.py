from enum import Enum


class TextObjectType(Enum):
    BASE_PDF = (".pdf",)
    CLEANED = "_cleaned.md"
    ORIGINAL_ABSTRACT = "_original_abstract.md"
    PRODUCED_ABSTRACT_ITERATIVE = "_produced_abstract_iterative.md"
    PRODUCED_ABSTRACT_MAP_REDUCE = "_produced_abstract_map_reduce.md"
    KEY_WORDS_STATISTICAL = "_key_words_statistical.md"
    KEY_WORDS_LLM = "_key_words_llm.md"
    KEY_WORDS_REFERENCE = "_key_words_reference.md"
