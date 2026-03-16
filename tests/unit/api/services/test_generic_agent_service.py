from unittest.mock import MagicMock, patch

import pytest
from langchain_core.messages import BaseMessage

from api.services.generic_agent_service import GenericAgentService

@pytest.mark.unit
@patch('api.services.generic_agent_service.LocalOllamaLLMWrapper')
@patch('api.services.generic_agent_service.ReactAgent')
def test_run_returns_agent_response(mock_react_agent_cls, mock_llm_wrapper_cls):
    mock_llm = MagicMock()
    mock_llm.get_model.return_value = 'fake-model'
    mock_llm_wrapper_cls.return_value = mock_llm

    mock_agent = MagicMock()
    mock_agent.invoke.return_value = {"reply": "ok"}
    mock_react_agent_cls.return_value = mock_agent

    service = GenericAgentService(name="test")
    messages = [MagicMock(spec=BaseMessage)]

    response = service.run(messages)

    assert response == {"reply": "ok"}
    mock_agent.invoke.assert_called_once_with(messages, {"configurable": {"thread_id":None}})