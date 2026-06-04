import pytest
import os
import hashlib
from src.phase4.image_gen import get_deterministic_seed

def test_deterministic_seed():
    shot_id = "shot_001"
    seed1 = get_deterministic_seed(shot_id)
    seed2 = get_deterministic_seed(shot_id)
    assert seed1 == seed2
    
    shot_id_2 = "shot_002"
    seed3 = get_deterministic_seed(shot_id_2)
    assert seed1 != seed3
    
    # Check if the logic matches what's required
    expected = int(hashlib.sha256(shot_id.encode()).hexdigest()[:8], 16)
    assert seed1 == expected

@pytest.mark.slow
def test_pipeline_load_skip():
    """Skip test for pipeline loading to avoid slow tests locally unless specified."""
    pass
