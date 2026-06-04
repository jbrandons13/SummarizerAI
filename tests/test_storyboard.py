import pytest
import json
import os
from unittest.mock import patch, MagicMock
from src.phase4.storyboard import extract_json, run_storyboard

def test_extract_json_clean():
    input_str = '[{"shot_id": "shot_01", "topic_tag": "test"}]'
    res = extract_json(input_str)
    assert len(res) == 1
    assert res[0]["shot_id"] == "shot_01"

def test_extract_json_markdown_block():
    input_str = '''Here is your json:
```json
[
  {"shot_id": "shot_02", "topic_tag": "markdown"}
]
```
Enjoy!
'''
    res = extract_json(input_str)
    assert len(res) == 1
    assert res[0]["topic_tag"] == "markdown"

def test_extract_json_trailing_comma():
    input_str = '''
[
  {"shot_id": "shot_03", "topic_tag": "comma"},
]
'''
    res = extract_json(input_str)
    assert len(res) == 1
    assert res[0]["topic_tag"] == "comma"

def test_extract_json_invalid_raises():
    with pytest.raises(json.JSONDecodeError):
        extract_json("not json at all")

@patch('src.phase4.storyboard.LocalBackend')
def test_run_storyboard_mocked(mock_backend_cls, tmp_path):
    video_id = "test_vid"
    video_dir = tmp_path / video_id
    phase4_dir = video_dir / "phase4"
    phase4_dir.mkdir(parents=True)
    
    # Create dummy summary
    summary = {
        "sentences": [
            {"id": "0", "keywords": ["rock", "stone"]},
            {"id": "1", "keywords": ["cycle"]}
        ]
    }
    with open(video_dir / "summary_script.json", "w") as f:
        json.dump(summary, f)
        
    # Create dummy shots
    shots = {
        "shots": [
            {"shot_id": "shot_001", "text": "This is a rock.", "source_segment_ids": ["0"]},
            {"shot_id": "shot_002", "text": "It is a cycle.", "source_segment_ids": ["1"]}
        ]
    }
    with open(phase4_dir / "shots.json", "w") as f:
        json.dump(shots, f)
        
    # Setup mock
    mock_instance = MagicMock()
    mock_backend_cls.return_value = mock_instance
    
    # Mock return value (batch of 2 shots)
    mock_response = '''```json
[
  {
    "shot_id": "shot_001",
    "visual_description": "A rock is shown.",
    "image_prompt": "A shiny rock",
    "key_entities": ["rock"],
    "topic_tag": "rock_formation"
  },
  {
    "shot_id": "shot_002",
    "visual_description": "A cycle.",
    "image_prompt": "A cycle, flat 2D educational cartoon style, clean lines, vivid colors",
    "key_entities": ["cycle"],
    "topic_tag": "rock_formation"
  }
]
```'''
    mock_instance.generate.return_value = mock_response
    mock_instance.model_name = "MockModel"
    
    out_path, fallbacks, tags = run_storyboard(video_id, str(tmp_path))
    
    assert os.path.exists(out_path)
    assert fallbacks == 0
    assert tags == {"rock_formation": 2}
    
    with open(out_path, "r") as f:
        data = json.load(f)
        
    assert "shots" in data
    assert "storyboards" not in data
    assert len(data["shots"]) == 2
    
    for shot in data["shots"]:
        assert "shot_id" in shot
        assert "visual_description" in shot
        assert "image_prompt" in shot
        assert "key_entities" in shot
        assert "topic_tag" in shot
        
        # Verify image prompt suffix
        assert shot["image_prompt"].endswith("flat 2D educational cartoon style, clean lines, vivid colors")
        assert shot["key_entities"] != ["fallback"]
