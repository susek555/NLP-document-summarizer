import os
from enum import Enum

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_groq import ChatGroq


class LLMEnum(Enum):
    LLAMA_GROK = "llama-3.3-70b-versatile"
    GEMINI_25_FLASH = "gemini-2.5-flash"

class LLMFactory:
    @staticmethod
    def get_llm(provider: LLMEnum) -> BaseChatModel:
        if provider == LLMEnum.LLAMA_GROK:
            return ChatGroq(
                model=provider.value,
                temperature=0,
                api_key=os.getenv("GROQ_API_KEY")
            )
        elif provider == LLMEnum.GEMINI_25_FLASH:
            return ChatGoogleGenerativeAI(
                model=provider.value,
                temperature=0,
                google_api_key=os.getenv("GOOGLE_API_KEY")
            )
