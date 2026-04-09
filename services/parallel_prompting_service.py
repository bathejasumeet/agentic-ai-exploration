from operator import add
from typing import TypedDict, Annotated

from langgraph.constants import START, END
from langgraph.graph import StateGraph

from core.llms.local_ollama_llm_wrapper import LocalOllamaLLMWrapper


class State(TypedDict):
    prompts: list[str]
    results: Annotated[list[str], add]
    final_result: str


class ParallelPromptingService:
    def __init__(self):
        self.llm = LocalOllamaLLMWrapper()

    def get_parallel_node(self, index):
        def node(state: State):
            result = self.llm.get_model().invoke(state['prompts'][index])
            return {
                'results': [result.content]
            }

        return node

    def aggregator(self, state: State):
        info = '\n'.join(f'info {i} : {result}' for i, result in enumerate(state['results']))
        final_result = self.llm.get_model().invoke(f"combine the information and summarize it in a funny way finding a link between them : {info}")
        return {
            'final_result': final_result
        }

    def create_parallel_prompting_graph(self, prompts_len: int):
        parallel_prompting_graph = StateGraph(State)
        parallel_prompting_graph.add_node('aggregator', self.aggregator)
        for i in range(0, prompts_len):
            node_name = f'parallel_node_{i}'
            parallel_prompting_graph.add_node(node_name, self.get_parallel_node(i))
            parallel_prompting_graph.add_edge(START, node_name)
            parallel_prompting_graph.add_edge(node_name, 'aggregator')
        parallel_prompting_graph.add_edge('aggregator', END)
        self.agent = parallel_prompting_graph.compile()

    def invoke(self, prompts: list[str]):
        self.create_parallel_prompting_graph(len(prompts))
        result = self.agent.invoke({
            'prompts': prompts,
            'results': []
        })
        return result

if __name__ == '__main__':
    prompts = [
        "Write a cat joke",
        "Write a dad fact",
        "Give me one healthy saying for mind"
    ]

    parallel_prompting_service = ParallelPromptingService()
    print(parallel_prompting_service.invoke(prompts))
