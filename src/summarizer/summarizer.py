from abc import ABC, abstractmethod

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_text_splitters import RecursiveCharacterTextSplitter


class Summarizer(ABC):
    def __init__(self, llm: BaseChatModel, chunk_size: int = 4000):
        self.llm = llm
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=400,
            separators=["\n#", "\n\n", "\n", " ", ""],
        )

    @abstractmethod
    def build_abstract(self, text: str) -> str:
        pass
