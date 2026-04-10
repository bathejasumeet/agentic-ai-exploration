from typing import Optional

from fastapi import APIRouter, Header, HTTPException, status
from pydantic import BaseModel

from services.parallel_prompting_service import ParallelPromptingService

parallel_prompting_service = ParallelPromptingService()


class ParallelProcessingRequest(BaseModel):
    prompts: list[str]


class ParallelProcessingResponse(BaseModel):
    response: str


router = APIRouter(prefix='/parallel-processing-agent')


@router.post('/chat')
def chat(
        request: ParallelProcessingRequest,
        x_client_id: Optional[str] = Header(None, alias='X-Client-Id')
) -> ParallelProcessingResponse:
    if not x_client_id:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail='X-Client-Id header absent'
        )
    result = parallel_prompting_service.invoke(request.prompts)['final_result']
    return ParallelProcessingResponse(
        response=result['messages'][-1].content
    )
