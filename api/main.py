import uvicorn
from fastapi import FastAPI
from langchain_core.messages import HumanMessage
from pydantic import BaseModel

from api.services.generic_agent_service import GenericAgentService

class GenricAgentResponse(BaseModel):
    response: str

app = FastAPI()
generic_agent_service = GenericAgentService(
    system_prompt="""
    You are a helpful assistant that can answer questions and perform tasks based on user input.
    You are allowed to be creative.
    Be friendly to the user, it is ok to crack jokes, but always be helpful and informative.""")


@app.post("/api/v1/agent-service/chat-agent/chat")
def chat(message: str) -> GenricAgentResponse:
    result = generic_agent_service.run([HumanMessage(content=message)])["messages"]
    return GenricAgentResponse(
        response=result[-1].content
    )
