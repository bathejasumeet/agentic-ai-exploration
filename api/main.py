from dotenv import load_dotenv

load_dotenv()

import uvicorn
from fastapi import FastAPI, APIRouter
from .routers.chat_agent import router as chat_agent_router
from .routers.prompt_chaining_agent import router as prompt_chaining_router
from .routers.parallel_processing_agent import router as parallel_processing_router
from .routers.orchestrator_synthesizer_agent import router as orchestrator_synthesizer_router

app = FastAPI()
router = APIRouter(prefix='/api/v1/agent-service')
router.include_router(chat_agent_router)
router.include_router(prompt_chaining_router)
router.include_router(parallel_processing_router)
router.include_router(orchestrator_synthesizer_router)

app.include_router(router)

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
