from langchain_core.messages import BaseMessage
from langgraph.typing import InputT

from core.agents.react_agent import ReactAgent
from core.llms.local_ollama_llm_wrapper import LocalOllamaLLMWrapper


class GenericAgentService:
    def __init__(self, system_prompt: str = None, name=None):
        self.llm = LocalOllamaLLMWrapper()
        self.agent = ReactAgent(self.llm.get_model(), system_prompt=system_prompt, name=name)

    def run(self, messages: list[BaseMessage], thread_id: str = None):
        return self.agent.invoke(messages, {"configurable": {"thread_id": thread_id}})