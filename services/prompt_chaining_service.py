from typing import Dict, Annotated

from langchain_core.messages import HumanMessage
from langgraph.constants import START, END
from langgraph.graph import MessagesState, StateGraph
from pydantic import Field

from core.llms.local_ollama_llm_wrapper import LocalOllamaLLMWrapper


class State(MessagesState):
    chains_count: int
    prompt: list[str]
    current_prompt_index: int


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
        self.agent = prompt_chainer_graph.compile()
        return self.agent

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
            'messages': [response],
            'current_prompt_index': index + 1
        }

    def invoke(self, initial_state: Dict):
        result = self.agent.invoke(initial_state)
        return result


if __name__ == '__main__':
    city_text = """
    * **Paris:** Started with a croissant & a breathtaking view of the Eiffel Tower. 🥐🗼
    * **Tokyo:** Neon lights, incredible ramen, and robots – seriously! 🍜🤖
    * **Rome:** Lost in the Colosseum, tossed a coin in the Trevi Fountain... pure history! 🏛️💦
    * **New York City:** The energy is *electric*! Broadway, Central Park, and endless possibilities. ✨
    * **Barcelona:** Gaudi's architecture is mind-blowing, plus amazing tapas! 🎨🍤
    * **London:** Red phone booths, Buckingham Palace, and a seriously cool pub scene. 📞👑🍻
    * **Rio de Janeiro:** Samba, beaches, and the Christ the Redeemer overlooking it all. 💃🏖️🙏
    * **Sydney:** Harbour Bridge selfies & Opera House vibes - iconic! 🌉🎶
    * **Buenos Aires:** Tango, steak, and a wonderfully passionate culture. 🔥🥩💃
    * **Dubai:** Seriously futuristic architecture & luxury shopping! 🇦🇪💎
    * **Marrakech:** Spicy smells, vibrant souks, & stunning riads. 🌶️🛍️🏡
    * **Istanbul:** A bridge between Europe & Asia, Hagia Sophia, & the Grand Bazaar. 🕌🌉
    * **Mexico City:** Ancient ruins, delicious street food, and a buzzing metropolis. 🌮🌋
    * **Cape Town:** Table Mountain views, beautiful beaches & a rich, diverse history. ⛰️🌊
    * **San Francisco:** Golden Gate Bridge, cable cars & a quirky, artistic vibe. 🌉🚋
    * **Shanghai:** Futuristic skyline, incredible food, & a super-efficient metro. 🏙️🍜🚇
    * **Berlin:** History, art, and a thriving music scene – totally cool. 🎶🎨
    * **Amsterdam:** Canals, bikes, and a relaxed, charming atmosphere. 🛶🚲
    * **Vienna:** Palaces, classical music, and amazing coffee! ☕️🎻
    * **Toronto:** Multicultural & vibrant, with amazing food & a beautiful waterfront. 🏙️🌊
    * **Singapore:** Futuristic architecture, incredible food & a super clean city! 🏙️🍜
    """

    prompt_chaining_service = PromptChainingService()
    prompt_chaining_service.invoke({
        'chains_count': 3,
        'prompt': [
            'extract cities from the following text text: ' + city_text,
            'describe cities in a single line',
            'add a story about a person in the city in three lines for each of the city'
        ],
        'current_prompt_index':0
    })
