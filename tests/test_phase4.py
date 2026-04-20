import pytest
import os
import json
from pathlib import Path
from src.phase4_retrieve import Phase4Retrieval
from src.schemas import SummaryScript, SummarySentence, RetrievalOutput
from src.utils.vram import VRAMManager

def test_phase4_full_pipeline():
    video_path = Path("tests/fixtures/tiny_video.mp4")
    if not video_path.exists():
        pytest.skip("tiny_video.mp4 not found in tests/fixtures/")
        
    output_dir = Path("data/intermediate/tiny_video")
    # Clean up previous runs
    if output_dir.exists():
        import shutil
        shutil.rmtree(output_dir)
        
    # Create a dummy SummaryScript
    summary = SummaryScript(
        video_id="tiny_video",
        target_duration=30,
        style="fast-paced",
        backend_used="dummy",
        sentences=[
            SummarySentence(
                id=0,
                text="The video starts with a close up of something.",
                estimated_duration_seconds=5.0,
                source_timestamp_hint=[0.0, 5.0],
                keywords=["start", "close up"]
            ),
            SummarySentence(
                id=1,
                text="Then we see some action happening in the middle.",
                estimated_duration_seconds=5.0,
                source_timestamp_hint=[10.0, 15.0],
                keywords=["action", "middle"]
            )
        ]
    )
    
    vram_manager = VRAMManager(device_id=0)
    orchestrator = Phase4Retrieval(vram_manager)
    
    results = orchestrator.run(video_path, summary, language="en")
    
    assert "random" in results
    assert "siglip_direct" in results
    assert "caption_cosine" in results
    
    for method, output in results.items():
        assert isinstance(output, RetrievalOutput)
        assert len(output.matches) == len(summary.sentences)
        
        # Check if files were saved
        out_file = output_dir / f"scene_matches_{method}.json"
        assert out_file.exists()
        
    # Arm B should have generated a cache
    cache_path = output_dir / "keyframes_captions.json"
    assert cache_path.exists()
    
    print("\n--- Retrieval Results Comparison ---")
    for i, sent in enumerate(summary.sentences):
        print(f"Sentence {i}: {sent.text}")
        for method in results:
            match = results[method].matches[i]
            print(f"  [{method}] Matched Scene {match.matched_scene_id} (Score: {match.score:.4f})")

if __name__ == "__main__":
    test_phase4_full_pipeline()
