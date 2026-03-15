from langchain_ollama import ChatOllama

from core.base.llm_wrapper import LlmWrapper


class LocalOllamaLLMWrapper(LlmWrapper):

    def __init__(self, model_name: str = "gemma3:1b-it-qat", base_url: str = "http://localhost:11434"):
        self.llm = ChatOllama(model=model_name, base_url=base_url)

    def get_model(self):
        return self.llm
