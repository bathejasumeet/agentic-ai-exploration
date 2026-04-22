import operator
from typing import TypedDict, Annotated, Dict

from langchain_core.messages import SystemMessage, HumanMessage
from langgraph.constants import START, END
from langgraph.graph import StateGraph
from langgraph.types import Send
from pydantic import BaseModel, Field

from core.llms.local_ollama_llm_wrapper import LocalOllamaLLMWrapper


class Section(BaseModel):
    name: str = Field(
        description='name for this section'
    )
    details: str = Field(
        description='details of the topic to be conveyed'
    )


class PlannerState(BaseModel):
    sections: list[Section] = Field(
        description='Sections of the report'
    )


class WorkerState(TypedDict):
    section: Section
    completed_sections: Annotated[list, operator.add]


class State(TypedDict):
    topic: str
    sections: list[Section]
    completed_sections: Annotated[list, operator.add]
    final_report: str


class OrchestratorSynthesizerService:
    def __init__(self):
        self.llm = LocalOllamaLLMWrapper().get_model()
        self.planner = self.llm.with_structured_output(PlannerState)
        self.worker = self.llm
        self.graph = self.create_graph()

    def create_graph(self):
        graph = StateGraph(State)
        graph.add_node('orchestrator', self.orchestrator)
        graph.add_node('synthesizer', self.synthesizer)
        graph.add_node('worker', self.worker)
        graph.add_edge(START, 'orchestrator')
        graph.add_conditional_edges('orchestrator', self.create_worker, 'worker')
        graph.add_edge('worker', 'synthesizer')
        graph.add_edge('synthesizer', END)
        return graph.compile()

    def orchestrator(self, state: State):
        """
        Orchestrator that plans splits the task into multiple subtasks
        """
        # Generate queries
        report_sections = self.planner.invoke(
            [
                SystemMessage(content="Generate a plan for the report."),
                HumanMessage(content=f"Here is the report topic: {state['topic']}"),
            ]
        )

        return {"sections": report_sections.sections}

    def worker(self, state: WorkerState):
        section = self.worker.invoke([
            SystemMessage(content="Generate a report on the topic and details that user provides. Make it informed"),
            HumanMessage(
                content=f"Generate a report on {state['section'].name} focusing on the details {state['section'].details}")
        ])
        return {'completed_sections': section}

    def synthesizer(self, state: State):
        """
        Synthesizer that combines the result of worker nodes working on sections
        """
        report = "-----".join(f"{i.name}  \n {i.details}" for i in state['sections'])

        return {"final_report": report}

    def create_worker(self, state: State):
        """Assign a worker to each section in the plan"""
        return [Send("worker", {"section": s}) for s in state["sections"]]

    def invoke(self, initial_state: Dict):
        result = self.graph.invoke(initial_state)
        return result

if __name__ == '__main__':
    orchestratorSynthesizerService = OrchestratorSynthesizerService()
    orchestratorSynthesizerService.invoke({
        "topic": "climate change"
    })
