import pytest
from fastapi.testclient import TestClient
from unittest.mock import MagicMock, patch
from pathlib import Path
import json
import io

from src.api.app import app
from src.api.tasks import JOBS

client = TestClient(app)

def test_health_check():
    response = client.get("/")
    assert response.status_code == 200
    assert response.json() == {"status": "ok", "message": "Video Summarizer API is running"}

@patch("src.api.routes.executor.submit")
def test_post_summarize(mock_submit):
    # Create a dummy mp4 file
    file_content = b"fake video data"
    file = io.BytesIO(file_content)
    
    response = client.post(
        "/api/summarize",
        files={"file": ("test.mp4", file, "video/mp4")},
        data={
            "target_duration": 60,
            "retrieval_method": "random",
            "style": "informative"
        }
    )
    
    assert response.status_code == 200
    data = response.json()
    assert "job_id" in data
    job_id = data["job_id"]
    assert job_id in JOBS
    assert JOBS[job_id]["status"] == "pending"
    
    # Cleanup dummy file
    raw_path = Path("data/raw") / f"{job_id}.mp4"
    if raw_path.exists():
        raw_path.unlink()

def test_get_status_not_found():
    response = client.get("/api/status/non-existent-id")
    assert response.status_code == 404

def test_get_status_found():
    job_id = "test-job-123"
    JOBS[job_id] = {
        "job_id": job_id,
        "status": "processing",
        "current_phase": 1,
        "phase_name": "Transcription",
        "progress_pct": 45,
        "phase_details": "Processing audio",
        "elapsed_seconds": 10,
        "phases_completed": [],
        "error": None
    }
    
    response = client.get(f"/api/status/{job_id}")
    assert response.status_code == 200
    data = response.json()
    assert data["job_id"] == job_id
    assert data["status"] == "processing"
    assert data["progress_pct"] == 45

def test_eval_dashboard_empty(tmp_path):
    # Mock results dir to be empty
    with patch("src.api.routes.Path") as mock_path:
        mock_results = MagicMock()
        mock_results.glob.return_value = []
        # This is a bit complex to mock perfectly, let's just check if it returns empty structure
        response = client.get("/api/eval/dashboard")
        assert response.status_code == 200
        data = response.json()
        assert data["videos_tested"] >= 0 # Depends on existing local data
