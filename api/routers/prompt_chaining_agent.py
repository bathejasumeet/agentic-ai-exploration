from typing import Optional

from fastapi import APIRouter, Header, HTTPException, status
from pydantic import BaseModel

from services.prompt_chaining_service import PromptChainingService

prompt_chaining_service=PromptChainingService()

class PromptChainingRequest(BaseModel):
    prompts: list[str]

class PromptChainingResponse(BaseModel):
    response: str

router = APIRouter(prefix='/prompt-chaining-agent')


@router.post('/chat')
def chat(
    request: PromptChainingRequest,
    x_client_id: Optional[str] = Header(None, alias="X-Client-Id"),
) -> PromptChainingResponse:
    if not x_client_id:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="X-Client-Id header absent",
        )
    result = prompt_chaining_service.invoke({
        'chains_count': len(request.prompts),
        'prompt': request.prompts,
        'current_prompt_index':0
    })
    return PromptChainingResponse(
        response=result['messages'][-1].content
    )
