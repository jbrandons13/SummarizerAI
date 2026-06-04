import pytest
import json
from src.phase4.anchor_policy import (
    AlwaysChainPolicy, NeverChainPolicy, FixedIntervalPolicy,
    SemanticTriggeredPolicy, AnchorDecision, serialize_decisions
)

@pytest.fixture
def synthetic_storyboard():
    return {
        "video_id": "test_vid",
        "shots": [
            {
                "shot_id": "shot_001",
                "visual_description": "A man is walking in the park.",
                "image_prompt": "man walking, park",
                "key_entities": "man, park",
                "topic_tag": "walking"
            },
            {
                "shot_id": "shot_002",
                "visual_description": "A man is taking a stroll in the park.",
                "image_prompt": "man strolling, park",
                "key_entities": "man, park",
                "topic_tag": "walking"
            },
            {
                "shot_id": "shot_003",
                "visual_description": "A completely different scene with a car racing.",
                "image_prompt": "car racing, fast",
                "key_entities": "car, track",
                "topic_tag": "racing"
            },
            {
                "shot_id": "shot_004",
                "visual_description": "Another car is seen on the track.",
                "image_prompt": "car, track",
                "key_entities": "car, track",
                "topic_tag": "racing"
            }
        ]
    }

def test_always_chain(synthetic_storyboard):
    policy = AlwaysChainPolicy()
    
    # Let's add one more shot to match the user's example
    synthetic_storyboard["shots"].append({
        "shot_id": "shot_005",
        "visual_description": "car parked",
        "topic_tag": "racing"
    })
    
    decisions = policy.resolve(synthetic_storyboard)
    assert len(decisions) == 5
    assert decisions[0].anchor_decision == "RESET"
    assert decisions[0].anchor_source is None
    
    # Explicitly check anchor_source is N-1
    assert decisions[1].anchor_source == "shot_001"
    assert decisions[2].anchor_source == "shot_002"
    assert decisions[3].anchor_source == "shot_003"
    assert decisions[4].anchor_source == "shot_004"

def test_never_chain(synthetic_storyboard):
    policy = NeverChainPolicy()
    decisions = policy.resolve(synthetic_storyboard)
    assert len(decisions) == 4
    for d in decisions:
        assert d.anchor_decision == "RESET"
        assert d.anchor_source is None

def test_fixed_interval():
    policy = FixedIntervalPolicy(k=5)
    storyboard = {"shots": [{"shot_id": f"shot_{i:03d}", "topic_tag": "tag"} for i in range(1, 12)]}
    decisions = policy.resolve(storyboard)
    assert len(decisions) == 11
    
    # 0-indexed: 0, 5, 10
    # shot_001, shot_006, shot_011
    
    assert decisions[0].anchor_decision == "RESET"
    assert decisions[0].anchor_source is None
    
    assert decisions[1].anchor_decision == "CHAIN"
    assert decisions[1].anchor_source == "shot_001"
    
    assert decisions[2].anchor_decision == "CHAIN"
    assert decisions[2].anchor_source == "shot_002"
    
    assert decisions[3].anchor_decision == "CHAIN"
    assert decisions[3].anchor_source == "shot_003"
    
    assert decisions[4].anchor_decision == "CHAIN"
    assert decisions[4].anchor_source == "shot_004"
    
    assert decisions[5].anchor_decision == "RESET"
    assert decisions[5].anchor_source is None
    
    assert decisions[6].anchor_decision == "CHAIN"
    assert decisions[6].anchor_source == "shot_006"
    
    assert decisions[10].anchor_decision == "RESET"
    assert decisions[10].anchor_source is None

@pytest.mark.slow
def test_semantic_triggered_real_model(synthetic_storyboard):
    policy = SemanticTriggeredPolicy(
        embedding_model="sentence-transformers/all-MiniLM-L6-v2",
        threshold_chain=0.75,
        threshold_soft=0.55
    )
    decisions = policy.resolve(synthetic_storyboard)
    
    assert len(decisions) == 4
    assert decisions[0].anchor_decision == "RESET"
    assert decisions[0].anchor_source is None
    
    # shot 2 has same topic tag, so it should have similarity calculated
    assert decisions[1].similarity_to_prev is not None
    # Depending on exact sim, it will be CHAIN or SOFT_CHAIN or RESET. 
    # But whatever it is, if it's CHAIN/SOFT_CHAIN, anchor_source should be shot N-1.
    if decisions[1].anchor_decision in ["CHAIN", "SOFT_CHAIN"]:
        assert decisions[1].anchor_source == "shot_001"
    else:
        assert decisions[1].anchor_source is None
    
    # shot 3 has different topic tag -> FORCE RESET, similarity is None
    assert decisions[2].anchor_decision == "RESET"
    assert decisions[2].similarity_to_prev is None
    assert decisions[2].anchor_source is None
    
    # shot 4 has same topic tag as shot 3, so similarity calculated
    assert decisions[3].similarity_to_prev is not None
    if decisions[3].anchor_decision in ["CHAIN", "SOFT_CHAIN"]:
        assert decisions[3].anchor_source == "shot_003"
    else:
        assert decisions[3].anchor_source is None

def test_semantic_triggered_boundaries(monkeypatch):
    policy = SemanticTriggeredPolicy(
        threshold_chain=0.75,
        threshold_soft=0.55
    )
    
    # We will mock the model's encode and util.pytorch_cos_sim instead of _compute_similarity
    class MockModel:
        def encode(self, texts, convert_to_tensor, show_progress_bar):
            # dummy embeddings, shape (len(texts), D)
            import torch
            return torch.zeros((len(texts), 1))
            
    monkeypatch.setattr(policy, "_load_model", lambda: MockModel())
    
    sim_values = [0.75, 0.55, 0.5499]
    sim_iter = iter(sim_values)
    
    import sentence_transformers.util
    def mock_cos_sim(emb1, emb2):
        class MockSim:
            def item(self):
                return next(sim_iter)
        return MockSim()
        
    monkeypatch.setattr(sentence_transformers.util, "pytorch_cos_sim", mock_cos_sim)
    
    storyboard = {
        "shots": [
            {"shot_id": "shot_001", "topic_tag": "A", "visual_description": "..."}
        ]
    }
    # Add shots that will trigger the 3 similarities
    for i in range(2, 5):
        storyboard["shots"].append({"shot_id": f"shot_{i:03d}", "topic_tag": "A", "visual_description": "..."})
        
    decisions = policy.resolve(storyboard)
    assert decisions[0].anchor_decision == "RESET"
    
    assert decisions[1].anchor_decision == "CHAIN"
    assert decisions[1].anchor_source == "shot_001"
    
    assert decisions[2].anchor_decision == "SOFT_CHAIN"
    assert decisions[2].anchor_source == "shot_002"
    
    assert decisions[3].anchor_decision == "RESET"
    assert decisions[3].anchor_source is None

def test_topic_tag_in_output():
    # Verifikasi decision.topic_tag selalu mencerminkan topic_tag shot itu sendiri, bukan diinherit.
    policy = AlwaysChainPolicy()
    storyboard = {
        "shots": [
            {"shot_id": "shot_001", "topic_tag": "A", "visual_description": "desc 1"},
            {"shot_id": "shot_002", "topic_tag": "B", "visual_description": "desc 2"}
        ]
    }
    decisions = policy.resolve(storyboard)
    
    assert len(decisions) == 2
    assert decisions[0].anchor_decision == "RESET"
    assert decisions[0].topic_tag == "A"
    
    # Walaupun CHAIN, dia tidak mewarisi "A" tapi harus "B"
    assert decisions[1].anchor_decision == "CHAIN"
    assert decisions[1].topic_tag == "B"

def test_single_shot():
    storyboard = {
        "shots": [
            {"shot_id": "shot_001", "topic_tag": "A", "visual_description": "..."}
        ]
    }
    for PolicyClass in [AlwaysChainPolicy, NeverChainPolicy, FixedIntervalPolicy, SemanticTriggeredPolicy]:
        policy = PolicyClass()
        decisions = policy.resolve(storyboard)
        assert len(decisions) == 1
        assert decisions[0].anchor_decision == "RESET"
        assert decisions[0].anchor_source is None

def test_all_different_topics():
    policy = SemanticTriggeredPolicy()
    storyboard = {
        "shots": [
            {"shot_id": f"shot_{i:03d}", "topic_tag": f"topic_{i}", "visual_description": "..."}
            for i in range(1, 5)
        ]
    }
    decisions = policy.resolve(storyboard)
    assert len(decisions) == 4
    for d in decisions:
        assert d.anchor_decision == "RESET"
        assert d.anchor_source is None
        if d.shot_id != "shot_001":
            assert d.similarity_to_prev is None

def test_output_schema_serializer():
    decisions = [
        AnchorDecision("shot_001", "RESET", None, None, "topic A")
    ]
    json_str = serialize_decisions(
        "vid_123", "semantic_triggered", {"threshold_chain": 0.75, "threshold_soft": 0.55}, decisions
    )
    data = json.loads(json_str)
    assert data["video_id"] == "vid_123"
    assert data["policy"] == "semantic_triggered"
    assert data["policy_config"]["threshold_chain"] == 0.75
    assert len(data["shots"]) == 1
    assert data["shots"][0]["shot_id"] == "shot_001"
    assert data["shots"][0]["anchor_decision"] == "RESET"
    assert data["shots"][0]["anchor_source"] is None
