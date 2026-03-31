from fastapi.testclient import  TestClient
from pydantic import ValidationError

from api.main import app
from api.routers.chat_agent import GenricAgentResponse

client = TestClient(app)

def test_chat_agent_unprocessable_entity():
    response = client.post('/api/v1/agent-service/chat-agent/chat')
    assert response.status_code == 422

def test_chat_agent_bad_request():
    response = client.post('/api/v1/agent-service/chat-agent/chat', json={
        "message":"Sample message"
    })
    assert response.status_code == 400

def test_chat_agent_success():
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

