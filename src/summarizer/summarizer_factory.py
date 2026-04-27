from enum import Enum

from langchain_core.language_models.chat_models import BaseChatModel

from src.summarizer.iterative_summarizer import IterativeSummarizer
from src.summarizer.map_reduce_summarizer import MapReduceSummarizer
from src.summarizer.summarizer import Summarizer


class SummarizerEnum(Enum):
    ITERATIVE = "iterative"
    MAP_REDUCE = "map_reduce"


class SummarizerFactory:
    @staticmethod
    def get_summarizer(
        type: SummarizerEnum, llm: BaseChatModel, chunk_size: int = 4000
    ) -> Summarizer:
        if type == SummarizerEnum.ITERATIVE:
            return IterativeSummarizer(llm, chunk_size)
        elif type == SummarizerEnum.MAP_REDUCE:
            return MapReduceSummarizer(llm, chunk_size)
