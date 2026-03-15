from typing import Any

from langchain.agents import create_agent
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import BaseMessage
from pydantic import BaseModel

from core.base.agent import BaseAgent


class MessagesInput(BaseModel):
    messages: list[BaseMessage]


class ReactAgent(BaseAgent):
    def __init__(self, llm: BaseChatModel, system_prompt: str = None, name=None):
        self.llm = llm
        self.agent = create_agent(
            model=llm,
            system_prompt=system_prompt,
            name=name,
            tools=[]
        )

    def invoke(self, messages: list[BaseMessage]) -> Any:
        return self.agent.invoke(MessagesInput(messages=messages))
