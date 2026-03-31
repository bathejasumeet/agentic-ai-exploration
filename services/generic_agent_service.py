from typing import Final

from langchain_core.messages import BaseMessage

from core.agents.react_agent import ReactAgent
from core.llms.local_ollama_llm_wrapper import LocalOllamaLLMWrapper


class GenericAgentService:
    SYSTEM_PROMPT: Final[str] = """
        You are a helpful assistant that can answer questions and perform tasks based on user input.
        You are allowed to be creative.
        Be friendly to the user, it is ok to crack jokes, but always be helpful and informative."""

    def __init__(self, name=None):
        self.llm = LocalOllamaLLMWrapper()
        self.agent = ReactAgent(self.llm.get_model(), system_prompt=self.SYSTEM_PROMPT, name=name)

    def run(self, messages: list[BaseMessage], thread_id: str = None):
        return self.agent.invoke(messages, {"configurable": {"thread_id": thread_id}})
