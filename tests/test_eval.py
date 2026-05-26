import pytest
import os
import json
from pathlib import Path
from unittest.mock import MagicMock, patch

from src.eval.metrics import compute_rouge, compute_bertscore, CLIPScoreCalculator
from src.eval.llm_judge import LLMJudge
from src.eval.run_ablation import AblationRunner

def test_compute_rouge():
    summary = "The cat sat on the mat."
    reference = "The cat sat on the mat."
    scores = compute_rouge(summary, reference)
    assert scores["rouge_l"] == 1.0

    summary2 = "A dog jumped over the fence."
    scores2 = compute_rouge(summary2, reference)
    assert scores2["rouge_l"] < 0.5

@pytest.mark.skipif(not os.path.exists("data/intermediate/tiny_video/keyframes/scene_000.jpg"), reason="Keyframe missing")
def test_clip_score():
    calculator = CLIPScoreCalculator()
    image_path = "data/intermediate/tiny_video/keyframes/scene_000.jpg"
    text = "A video frame"
    score = calculator.compute(image_path, text)
    assert score > 0
    assert isinstance(score, float)

@patch("src.models.llm_wrapper.GroqBackend.generate")
def test_llm_judge_mock(mock_generate):
    # Mock response
    mock_generate.return_value = json.dumps({
        "information_retention": 5,
        "factual_faithfulness": 4,
        "visual_relevance": 5,
        "reasoning": "Excellent summary."
    })
    
    with patch.dict(os.environ, {"GROQ_API_KEY": "fake"}):
        judge = LLMJudge()
        result = judge.evaluate_video("Transcript...", "Summary...", "Captions...")
    
    assert result["information_retention"] == 5
    assert result["factual_faithfulness"] == 4
    assert result["visual_relevance"] == 5
    assert "reasoning" in result

@pytest.mark.skipif(not os.path.exists("tests/fixtures/tiny_video.mp4"), reason="Fixtures missing")
@patch("src.models.llm_wrapper.GroqBackend.generate")
def test_eval_end_to_end_mocked(mock_generate):
    # Mock response for Groq LLM backend
    mock_generate.return_value = json.dumps({
        "video_id": "tiny_video",
        "target_duration": 60,
        "style": "informative",
        "backend_used": "groq",
        "sentences": [
            {
                "id": 0,
                "text": "The final joke about balloons is very funny.",
                "estimated_duration_seconds": 5.0,
                "source_timestamp_hint": [0.0, 5.0],
                "keywords": ["balloons", "funny joke"]
            }
        ]
    })
    
    # Mock judge and metrics to avoid long runs/costs
    with patch("src.eval.llm_judge.LLMJudge.evaluate_video") as mock_judge, \
         patch("src.eval.llm_judge.LLMJudge.__init__", return_value=None), \
         patch("src.eval.metrics.CLIPScoreCalculator.compute") as mock_clip:
        
        mock_judge.return_value = {
            "information_retention": 5, "factual_faithfulness": 5, 
            "visual_relevance": 5, "reasoning": "Mocked."
        }
        mock_clip.return_value = 25.0
        
        config = {
            "paths": {
                "intermediate_dir": "data/intermediate",
                "results_dir": "results_test"
            },
            # Dummy configs for pipeline
            "llm": {"backend": "groq", "groq": {"model_name": "llama-3.3-70b-versatile"}},
            "tts": {"backend": "kokoro"},
            "vram": {"device_id": 0, "limit_gb": 22.0}
        }
        
        # Ensure results_test exists
        Path("results_test").mkdir(exist_ok=True)
        
        runner = AblationRunner(config)
        # Initialize the mock and its backend manually since we mocked __init__
        runner.judge = MagicMock()
        runner.judge.evaluate_video = mock_judge
        runner.judge.get_cost_estimate.return_value = 0.0
        
        video_path = Path("tests/fixtures/tiny_video.mp4")
        # Only run random arm for speed
        results_dir, _ = runner.run([video_path], ["random"])
        
        assert results_dir.exists()
        assert (results_dir / "ablation_results.csv").exists()
        assert (results_dir / "summary.md").exists()
        assert (results_dir / "plots.png").exists()
