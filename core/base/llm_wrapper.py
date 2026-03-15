import abc

from langchain_core.language_models import BaseChatModel
from langchain_ollama import ChatOllama


class LlmWrapper(abc.ABC):

    @abc.abstractmethod
    def get_model(self) -> BaseChatModel:
        pass