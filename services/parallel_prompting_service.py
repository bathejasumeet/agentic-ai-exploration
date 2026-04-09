from operator import add
from typing import TypedDict, Annotated

from langgraph.constants import START, END
from langgraph.graph import StateGraph

from core.llms.local_ollama_llm_wrapper import LocalOllamaLLMWrapper


class State(TypedDict):
    prompts: list[str]
    results: Annotated[list[str], add]
    len: int


class ParallelPromptingService:
    def __init__(self):
        self.llm = LocalOllamaLLMWrapper()

    def get_parallel_node(self, index):
        def node(state: State):
            result = self.llm.get_model().invoke(state['prompts'][index])
            return {
                'results': [result]
            }
        return node

    def aggregator(self, state: State):
        info = '\n'.join(f'info {i} : {result}' for i, result in enumerate(state['results']))
        final_result = self.llm.get_model().invoke(f"combine the information and summarize it : {info}")
        return final_result

    def create_parallel_prompting_graph(self, prompts: list[str]):
        parallel_prompting_graph = StateGraph(State)
        parallel_prompting_graph.add_edge('aggregator', self.aggregator)
        for i,prompt in enumerate(prompts):
            node_name = f'parallel_node_{i}'
            parallel_prompting_graph.add_node(node_name, self.get_parallel_node(i))
            parallel_prompting_graph.add_edge(START, node_name)
            parallel_prompting_graph.add_edge(node_name, 'aggregator')
        parallel_prompting_graph.add_edge('aggregator', END)

