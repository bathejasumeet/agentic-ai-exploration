from unittest.mock import Mock

from fastapi.testclient import  TestClient
from langchain_core.language_models import GenericFakeChatModel
from langchain_core.messages import AIMessage
from pydantic import ValidationError

from api.main import app
from api.routers import chat_agent
from api.routers.chat_agent import GenricAgentResponse


def test_chat_agent_unprocessable_entity():
    client = TestClient(app)
    response = client.post('/api/v1/agent-service/chat-agent/chat')
    assert response.status_code == 422

def test_chat_agent_bad_request():
    client = TestClient(app)
    response = client.post('/api/v1/agent-service/chat-agent/chat', json={
        "message":"Sample message"
    })
    assert response.status_code == 400

def test_chat_agent_success(monkeypatch):
    client = TestClient(app)

    def fake_invoke(self, *args, **kwargs):
        return {"messages": [AIMessage(content="Sample Message")]}

    monkeypatch.setattr(chat_agent.generic_agent_service.agent,
                        "invoke",
                        fake_invoke.__get__(chat_agent.generic_agent_service.agent))


    response = client.post('/api/v1/agent-service/chat-agent/chat',
                           headers={
                               "X-Client-Id":"test-client-id"
                           },
                           json={
        "message":"Sample message"
    })
    assert response.status_code == 200
    try:
        GenricAgentResponse(**response.json())
    except ValidationError as e:
        assert False, f"Response validation failed: {e}"

