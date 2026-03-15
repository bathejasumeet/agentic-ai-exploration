from typing import Any

from langchain.agents import create_agent
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import BaseMessage
from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.memory import InMemorySaver

from core.base.agent import BaseAgent, MessagesInput


class ReactAgent(BaseAgent):
    def __init__(self, llm: BaseChatModel, system_prompt: str = None, name=None):
        self.llm = llm
        self.agent = create_agent(
            model=llm,
            system_prompt=system_prompt,
            checkpointer=InMemorySaver(),
            name=name,
            tools=[]
        )

    def invoke(self, messages: list[BaseMessage], configurable_input: RunnableConfig) -> Any:
        return self.agent.invoke(MessagesInput(messages=messages), config=configurable_input)
