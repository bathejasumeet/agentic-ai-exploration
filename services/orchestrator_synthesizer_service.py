import operator
from typing import TypedDict, Annotated

from langchain_core.messages import SystemMessage, HumanMessage
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
            HumanMessage(content=f"Generate a report on {state['section'].name} focusing on the details {state['section'].details}")
        ])
        return {'completed_sections': section}

    def synthesizer(self, state: State):
        pass

    def create_worker(self, state: State):
        pass
