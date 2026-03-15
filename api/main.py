from typing import Optional
import uvicorn

from langchain_core.messages import HumanMessage
from pydantic import BaseModel
from fastapi import FastAPI, Header, HTTPException, status

from api.services.generic_agent_service import GenericAgentService

class GenricAgentResponse(BaseModel):
    response: str

class GenericAgentRequest(BaseModel):
    message: str

app = FastAPI()
generic_agent_service = GenericAgentService(
    system_prompt="""
    You are a helpful assistant that can answer questions and perform tasks based on user input.
    You are allowed to be creative.
    Be friendly to the user, it is ok to crack jokes, but always be helpful and informative.""")


@app.post("/api/v1/agent-service/chat-agent/chat")
def chat(
    message: GenericAgentRequest,
    x_client_id: Optional[str] = Header(None, alias="X-Client-Id"),
) -> GenricAgentResponse:
    if not x_client_id:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="X-Client-Id header absent",
        )
    result = generic_agent_service.run([HumanMessage(content=message.message)], thread_id=x_client_id)["messages"]
    return GenricAgentResponse(
        response=result[-1].content
    )

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)