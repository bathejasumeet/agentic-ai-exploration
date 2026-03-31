from typing import Optional

from fastapi import Header, HTTPException, status, APIRouter
from langchain_core.messages import HumanMessage
from pydantic import BaseModel

from services.generic_agent_service import GenericAgentService

router = APIRouter(prefix='/chat-agent')
generic_agent_service = GenericAgentService()

class GenricAgentResponse(BaseModel):
    response: str

class GenericAgentRequest(BaseModel):
    message: str


@router.post("/api/v1/agent-service/chat-agent/chat")
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
