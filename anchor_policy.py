import json
import random
import re
from typing import Dict, List, Optional, Protocol, Any
from dataclasses import dataclass, asdict
import numpy as np

@dataclass
class AnchorDecision:
    shot_id: str
    anchor_decision: str
    anchor_source: Optional[str]
    similarity_to_prev: Optional[float]
    topic_tag: str
    # --- NEW (trailing, default None -> backward compatible with existing policies) ---
    anchor_concept: Optional[str] = None      # which recurring concept drove the anchor
    reference_distance: Optional[int] = None  # how many shots back the reference is

class AnchorPolicy(Protocol):
    @property
    def name(self) -> str:
        ...
    def resolve(self, storyboard: Dict[str, Any]) -> List[AnchorDecision]:
        ...

# ======================= EXISTING POLICIES (unchanged) =======================
class AlwaysChainPolicy:
    name = "always_chain"
    def resolve(self, storyboard: Dict[str, Any]) -> List[AnchorDecision]:
        shots = storyboard.get("shots", [])
        decisions = []
        for i, shot in enumerate(shots):
            shot_id = shot["shot_id"]; current_topic = shot.get("topic_tag", "")
            if i == 0:
                decisions.append(AnchorDecision(shot_id, "RESET", None, None, current_topic))
            else:
                decisions.append(AnchorDecision(shot_id, "CHAIN", shots[i-1]["shot_id"], None, current_topic))
        return decisions

class NeverChainPolicy:
    name = "never_chain"
    def resolve(self, storyboard: Dict[str, Any]) -> List[AnchorDecision]:
        shots = storyboard.get("shots", [])
        return [AnchorDecision(s["shot_id"], "RESET", None, None, s.get("topic_tag", "")) for s in shots]

class FixedIntervalPolicy:
    name = "fixed_interval"
    def __init__(self, k: int = 5):
        self.k = k
    def resolve(self, storyboard: Dict[str, Any]) -> List[AnchorDecision]:
        shots = storyboard.get("shots", [])
        decisions = []
        for i, shot in enumerate(shots):
            shot_id = shot["shot_id"]; current_topic = shot.get("topic_tag", "")
            if i % self.k == 0:
                decisions.append(AnchorDecision(shot_id, "RESET", None, None, current_topic))
            else:
                decisions.append(AnchorDecision(shot_id, "CHAIN", shots[i-1]["shot_id"], None, current_topic))
        return decisions

class SemanticTriggeredPolicy:
    name = "semantic_triggered"
    def __init__(self, embedding_model="sentence-transformers/all-MiniLM-L6-v2",
                 threshold_chain=0.75, threshold_soft=0.55):
        self.embedding_model = embedding_model
        self.threshold_chain = threshold_chain; self.threshold_soft = threshold_soft
        self._model = None
    def _load_model(self):
        if self._model is None:
            import torch
            from sentence_transformers import SentenceTransformer
            torch.manual_seed(42); np.random.seed(42); random.seed(42)
            self._model = SentenceTransformer(self.embedding_model)
        return self._model
    def _get_text_for_embedding(self, shot): return shot.get("visual_description", "")
    def resolve(self, storyboard):
        shots = storyboard.get("shots", [])
        if not shots: return []
        model = self._load_model()
        texts = [self._get_text_for_embedding(s) for s in shots]
        import torch
        with torch.no_grad():
            embeddings = model.encode(texts, convert_to_tensor=True, show_progress_bar=False)
        decisions = []
        from sentence_transformers import util
        for i, shot in enumerate(shots):
            shot_id = shot["shot_id"]; current_topic = shot.get("topic_tag", "")
            if i == 0:
                decisions.append(AnchorDecision(shot_id, "RESET", None, None, current_topic)); continue
            prev_topic = shots[i-1].get("topic_tag", "")
            if current_topic != prev_topic:
                decisions.append(AnchorDecision(shot_id, "RESET", None, None, current_topic)); continue
            sim = float(util.pytorch_cos_sim(embeddings[i], embeddings[i-1]).item())
            if sim >= self.threshold_chain:
                decisions.append(AnchorDecision(shot_id, "CHAIN", shots[i-1]["shot_id"], sim, current_topic))
            elif self.threshold_soft <= sim < self.threshold_chain:
                decisions.append(AnchorDecision(shot_id, "SOFT_CHAIN", shots[i-1]["shot_id"], sim, current_topic))
            else:
                decisions.append(AnchorDecision(shot_id, "RESET", None, sim, current_topic))
        return decisions

# ======================= NEW: free baseline =======================
class SegmentBoundaryPolicy:
    """FREE null-hypothesis baseline: ANCHOR to N-1 while inside the same source
    sentence, RESET at each new sentence. Zero embeddings. This is what the data
    showed semantic_triggered mostly recovers (~89% binary agreement); keeping it
    as an explicit baseline is what makes the thesis comparison honest."""
    name = "segment_boundary"
    def resolve(self, storyboard, shots_index=None):
        shots = storyboard.get("shots", [])
        decisions = []
        prev_seg = None
        for i, shot in enumerate(shots):
            shot_id = shot["shot_id"]; topic = shot.get("topic_tag", "")
            seg = shot.get("_segment_id")  # injected by runner from shots.json
            if i == 0 or seg != prev_seg or seg is None:
                decisions.append(AnchorDecision(shot_id, "RESET", None, None, topic))
            else:
                decisions.append(AnchorDecision(shot_id, "CHAIN", shots[i-1]["shot_id"], None, topic))
            prev_seg = seg
        return decisions

# ======================= NEW: PROPOSED contribution =======================
def _normalize_concept(e: str) -> str:
    e = e.lower().strip()
    e = re.sub(r"[^a-z0-9 ]", "", e)
    e = re.sub(r"\s+", " ", e)
    return e

class ConceptAnchorPolicy:
    """PROPOSED. Enforce visual consistency of RECURRING CONCEPTS across the whole
    summary (incl. non-adjacent recurrences) by conditioning generation on a
    per-concept reference image.

    For shot N: if any of its key_entities already appeared in an earlier shot,
    ANCHOR (IP-Adapter at image stage) to that concept's reference; else RESET.
    The 'driving' concept = the shared concept with the highest GLOBAL recurrence
    (ties -> earliest canonical). topic_tag is deliberately NOT used (that is what
    collapsed the old policy to a sentence detector).

    reference:
      'canonical' -> first occurrence of the driving concept (stable global identity)
      'recent'    -> most recent prior occurrence (smoother local drift)
    """
    def __init__(self, reference: str = "canonical", match: str = "exact"):
        assert reference in ("canonical", "recent")
        self.reference = reference; self.match = match
        self.name = f"concept_anchor_{reference}"

    def _concepts(self, shot):
        return set(c for c in (_normalize_concept(e) for e in shot.get("key_entities", [])) if c)

    def resolve(self, storyboard):
        shots = storyboard.get("shots", [])
        if not shots: return []
        concept_sets = [self._concepts(s) for s in shots]
        # global frequency of each concept (how many shots mention it at all)
        freq = {}
        for cs in concept_sets:
            for c in cs: freq[c] = freq.get(c, 0) + 1
        # occurrences (shot indices) per concept, in order
        occ = {}
        for i, cs in enumerate(concept_sets):
            for c in cs: occ.setdefault(c, []).append(i)

        decisions = []
        for i, shot in enumerate(shots):
            shot_id = shot["shot_id"]; topic = shot.get("topic_tag", "")
            if i == 0:
                decisions.append(AnchorDecision(shot_id, "RESET", None, None, topic)); continue
            # concepts of this shot that have a PRIOR occurrence
            shared = [c for c in concept_sets[i] if any(j < i for j in occ.get(c, []))]
            if not shared:
                decisions.append(AnchorDecision(shot_id, "RESET", None, None, topic)); continue
            # driving concept: highest global frequency, tie-break by earliest canonical
            driving = sorted(shared, key=lambda c: (-freq[c], occ[c][0]))[0]
            priors = [j for j in occ[driving] if j < i]
            ref_idx = min(priors) if self.reference == "canonical" else max(priors)
            decisions.append(AnchorDecision(
                shot_id, "CONCEPT_ANCHOR", shots[ref_idx]["shot_id"], None, topic,
                anchor_concept=driving, reference_distance=i - ref_idx))
        return decisions

def serialize_decisions(video_id, policy_name, policy_config, decisions):
    return json.dumps({"video_id": video_id, "policy": policy_name,
                       "policy_config": policy_config,
                       "shots": [asdict(d) for d in decisions]}, indent=2)
