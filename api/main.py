import uvicorn
from fastapi import FastAPI, APIRouter
from .routers.chat_agent import router as chat_agent_router


app = FastAPI()
router = APIRouter(prefix='/api/v1/agent-service')
router.include_router(chat_agent_router)


app.include_router(router)


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
