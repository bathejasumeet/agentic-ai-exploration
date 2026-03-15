import abc
from typing import Any

from langchain_core.messages import BaseMessage
from langchain_core.runnables import RunnableConfig
from pydantic import BaseModel


class MessagesInput(BaseModel):
    messages: list[BaseMessage]


class BaseAgent(abc.ABC):
    """
       Abstract base class for agents.
       Subclasses must implement the act method.
       """

    @abc.abstractmethod
    def invoke(self, messages: list[BaseMessage], configurable_input: RunnableConfig) -> Any:
        pass
