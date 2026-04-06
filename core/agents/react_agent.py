import os
from typing import Any

import psycopg
from anyio.lowlevel import checkpoint
from langchain.agents import create_agent
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import BaseMessage
from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.checkpoint.postgres import PostgresSaver

from core.base.agent import BaseAgent, MessagesInput


class ReactAgent(BaseAgent):
    def __init__(self, llm: BaseChatModel, system_prompt: str = None, name=None):
        self.llm = llm
        self._conn = None
        if "POSTGRES_URI_CUSTOM" in os.environ:
            self._conn = psycopg.connect(os.environ["POSTGRES_URI_CUSTOM"], autocommit=True)
            checkpointer = PostgresSaver(self._conn)
            checkpointer.setup()
            self.agent = create_agent(
                model=llm,
                system_prompt=system_prompt,
                checkpointer=checkpointer,
                name=name,
                tools=[]
            )

        else:
            self.agent = create_agent(
                model=llm,
                system_prompt=system_prompt,
                checkpointer=InMemorySaver(),
                name=name,
                tools=[]
            )


    def invoke(self, messages: list[BaseMessage], configurable_input: RunnableConfig) -> Any:
        return self.agent.invoke(MessagesInput(messages=messages), config=configurable_input)

    def __del__(self):
        if self._conn and not self._conn.closed:
            self._conn.close()