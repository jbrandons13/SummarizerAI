import pytest
from unittest.mock import patch, MagicMock
from src.phase4.comfyui_client import ComfyUIClient

@patch("urllib.request.urlopen")
def test_comfyui_client_system_stats(mock_urlopen):
    mock_response = MagicMock()
    mock_response.read.return_value = b'{"system": {"os": "linux"}}'
    mock_urlopen.return_value.__enter__.return_value = mock_response
    
    client = ComfyUIClient()
    stats = client.system_stats()
    assert stats == {"system": {"os": "linux"}}

@patch("urllib.request.urlopen")
def test_comfyui_client_queue_workflow(mock_urlopen):
    mock_response = MagicMock()
    mock_response.read.return_value = b'{"prompt_id": "12345"}'
    mock_urlopen.return_value.__enter__.return_value = mock_response
    
    client = ComfyUIClient()
    pid = client.queue_workflow({"1": {"class_type": "Dummy"}})
    assert pid == "12345"

@patch("websocket.WebSocket")
def test_comfyui_client_wait_for_completion(mock_ws_class):
    mock_ws = MagicMock()
    mock_ws_class.return_value = mock_ws
    
    mock_ws.recv.side_effect = [
        '{"type": "executing", "data": {"node": "1", "prompt_id": "12345"}}',
        '{"type": "executing", "data": {"node": null, "prompt_id": "12345"}}'
    ]
    
    client = ComfyUIClient()
    res = client.wait_for_completion("12345")
    assert res["status"] == "success"

@patch("websocket.WebSocket")
def test_comfyui_client_wait_for_completion_error(mock_ws_class):
    mock_ws = MagicMock()
    mock_ws_class.return_value = mock_ws
    
    mock_ws.recv.side_effect = [
        '{"type": "execution_error", "data": {"exception_type": "RuntimeError", "exception_message": "CUDA out of memory", "node_id": "1", "traceback": ""}}'
    ]
    
    client = ComfyUIClient()
    res = client.wait_for_completion("12345")
    assert res["status"] == "oom"
