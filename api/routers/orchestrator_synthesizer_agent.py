from typing import Optional

from fastapi import APIRouter, Header, HTTPException, status
from pydantic import BaseModel

from services.orchestrator_synthesizer_service import OrchestratorSynthesizerService

orchestrator_synthesizer_service = OrchestratorSynthesizerService()


class OrchestratorSynthesizerRequest(BaseModel):
    topic: str


class OrchestratorSynthesizerResponse(BaseModel):
    response: str


router = APIRouter(prefix='/orchestrator_synthesizer_agent')


@router.post('/chat')
def chat(
        request: OrchestratorSynthesizerRequest,
        x_client_id: Optional[str] = Header(None, alias='X-Client-Id')
) -> OrchestratorSynthesizerResponse:
    if not x_client_id:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail='X-Client-Id header absent'
        )
    result = orchestrator_synthesizer_service.invoke(request.topic)['final_report']
    return OrchestratorSynthesizerResponse(
        response=result
    )
