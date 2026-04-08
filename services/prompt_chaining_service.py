from pydantic import Field
from typing import Dict, Annotated

from langchain_core.messages import HumanMessage
from langgraph.constants import START, END
from langgraph.graph import MessagesState, StateGraph

from core.llms.local_ollama_llm_wrapper import LocalOllamaLLMWrapper


class State(MessagesState):
    chains_count: Annotated[int, Field(default=1)]
    prompt: list[str]
    current_prompt_index: Annotated[int, Field(default=0)]


class PromptChainingService:
    def __init__(self):
        self.llm = LocalOllamaLLMWrapper()
        self._prompt_chaining_graph = self._create_prompt_chaining_graph()

    def _create_prompt_chaining_graph(self):
        prompt_chainer_graph = StateGraph(State)
        prompt_chainer_graph.add_node('llm_call', self._llm_call)
        prompt_chainer_graph.add_edge(START, 'llm_call')
        prompt_chainer_graph.add_conditional_edges('llm_call', self._should_prompt,
                                                   {True: "llm_call", False: END})
        prompt_chainer_graph.compile()
        return prompt_chainer_graph

    @staticmethod
    def _should_prompt(state: State):
        if state['current_prompt_index'] < state['chains_count']:
            return True
        return False

    def _llm_call(self, state: State):
        index = state['current_prompt_index']
        prompt_text = state['prompt'][index]
        response = self.llm.get_model().invoke(state['messages'] + [HumanMessage(content=prompt_text)])
        return {
            'messages' : [response],
            'current_prompt_index' : index + 1
        }

    def invoke(self, initial_state: Dict):
        result = self._prompt_chaining_graph.invoke(initial_state)
        return result

if __name__ == '__main__':
    prompt_chaining_service = PromptChainingService()
    prompt_chaining_service.invoke({
        'chains_count':3,
        'prompt': [
            'extract cities from the provided text',
            'describe cities in a single line',
            'add a story about a person in the city in three lines for each of the city'
        ]
    })

