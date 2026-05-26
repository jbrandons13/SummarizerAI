import pytest
import os
import yaml
from pathlib import Path
from src.pipeline import VideoSummarizerPipeline
from src.schemas import Phase5Output

@pytest.mark.integration
def test_full_pipeline_integration(tmp_path):
    """
    End-to-end integration test for the entire pipeline.
    This test runs the actual models and takes time/VRAM.
    """
    # 1. Setup config
    with open("configs/default.yaml", "r") as f:
        config = yaml.safe_load(f)
    
    # Override paths to use tmp_path
    config["paths"]["intermediate_dir"] = str(tmp_path / "intermediate")
    config["paths"]["output_dir"] = str(tmp_path / "output")
    
    # Ensure intermediate and output dirs are set in config root if used by assembler
    config["intermediate_dir"] = config["paths"]["intermediate_dir"]
    config["output_dir"] = config["paths"]["output_dir"]
    
    # 2. Run Pipeline
    # Use tiny_video.mp4 for integration test
    video_path = Path("tests/fixtures/tiny_video.mp4")
    
    # Use manual .env loader if python-dotenv is missing
    if not os.getenv("GROQ_API_KEY"):
        env_path = Path(".env")
        if env_path.exists():
            with open(env_path, "r") as f:
                for line in f:
                    if line.strip() and not line.startswith("#"):
                        key, _, value = line.partition("=")
                        if key.strip() == "GROQ_API_KEY":
                            os.environ["GROQ_API_KEY"] = value.strip()
        
    if not os.getenv("GROQ_API_KEY") and config.get("llm", {}).get("backend") == "groq":
        pytest.skip("GROQ_API_KEY not found in environment. Skipping integration test.")

    pipeline = VideoSummarizerPipeline(config)
    output = pipeline.run(video_path, method="grouping_gate")
    
    # 3. Assertions
    assert isinstance(output, Phase5Output)
    assert Path(output.output_path).exists()
    assert output.video_id == "tiny_video"
    assert output.method == "grouping_gate"
    assert output.total_duration_seconds > 0
    
    # Verify metadata
    metadata_json = Path(config["output_dir"]) / "tiny_video" / "summary_grouping_gate_metadata.json"
    assert metadata_json.exists()
    
    print(f"Integration test passed. Summary video at: {output.output_path}")
