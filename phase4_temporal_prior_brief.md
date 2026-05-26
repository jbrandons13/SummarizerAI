# Phase 4 Update — Temporal Prior + Threshold Tuning

## Role and rules of engagement

This is a focused update to the Phase 4 retrieval gate. The algorithm now applies a Gaussian temporal prior to scene similarity, addressing the issue we identified where the global search was matching narration to visually similar scenes located far from the sentence's `source_timestamp_hint`.

The new file content and the config update are hard-coded below. Apply verbatim, do not redesign.

**Behavior expectations:**

- Be direct. No filler.
- After every task, print `=== TASK N DONE ===` to stdout.
- If a task is blocked, log the blocker and continue with non-blocked tasks.
- Do not commit or push.

## What changed and why

Previously, the gate did a global argmax over scene embeddings using raw cosine similarity. This caused matches like: sentence 0 (hint at second 12) selected scene 80 (at minute 9) because the raw cosine was marginally higher there.

The new gate scores each candidate scene as:

```
weighted_sim = cosine(text_emb, scene_emb) * gaussian_weight(scene_center_time, hint_center_time, sigma)
```

The Gaussian weight is 1.0 when the scene's centre coincides with the hint centre and decays with distance (≈0.61 at one sigma, ≈0.14 at two sigma, ≈0.01 at three sigma). The weight never reaches zero, so a visually overwhelming match far away can still win, but it is heavily suppressed.

The decision gate now applies to the weighted similarity. Threshold is lowered from `0.25` to `0.13` to fit the SigLIP 2 raw cosine range we confirmed in diagnostics (raw cosines hover at `0.10–0.18` for valid matches).

## Task overview

| Task | Description | Blocking next? |
|---|---|---|
| 1 | Replace `src/phase4_retrieve.py` with new code (hard-coded below) | Yes |
| 2 | Update `configs/default.yaml` Phase 4 block | Yes |
| 3 | Update the smoke-test print block in `src/pipeline.py` to surface new fields | No |
| 4 | Run smoke test on `review_2.mp4` and report | Yes |

Execute in order 1 → 2 → 3 → 4.

---

## Task 1: Replace `src/phase4_retrieve.py`

Overwrite `src/phase4_retrieve.py` with the exact content below. No edits.

```python
"""Phase 4: Grouping-based retrieval with decision gate and temporal prior.

Replaces the prior CCMA / DP / Hungarian alignment approach. The new design:

1. Walk the narration sentences forward and greedily form groups of consecutive
   sentences that all map best to the same scene in the source video.
2. For each group, decide whether the best matching scene is similar enough to
   be retrieved as-is, or whether the similarity is too weak and the group
   should be routed to Phase 5 (image-to-video generation).

To prevent grouping from matching narration to visually similar but temporally
distant scenes (e.g. a sentence about the start of the video matching a scene
near the end), a Gaussian temporal prior is applied. Each scene's contribution
to a sentence's score is multiplied by a weight that decays with the distance
between the scene's centre time and the sentence's source_timestamp_hint
centre. This preserves the CCMA-era assumption that narration tends to follow
the source video's temporal order while still allowing the model to select
visually compelling scenes nearby.

Two outputs feed Phase 5 downstream:
- the locked scene id (used for frame selection inside that scene)
- the merged timestamp range of the group (used to narrow the frame window)

The text encoder is treated as an injected dependency. The expected interface is
a callable ``encode(text: str) -> np.ndarray`` returning a single 1-D embedding.
This matches a thin wrapper around the existing SigLIP text encoder used to
embed scenes elsewhere in the pipeline.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, List, Optional, Protocol, Sequence, Tuple

import numpy as np


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------


class TextEncoder(Protocol):
    """Minimal interface a text encoder must satisfy.

    The implementation is expected to share the same embedding space as the
    scene encoder used during preprocessing. For SigLIP this means using the
    SigLIP text tower with the same model id as the image tower that produced
    the scene embeddings.
    """

    def encode(self, text: str) -> np.ndarray:  # pragma: no cover - protocol
        ...


@dataclass
class Sentence:
    """One narration sentence from Phase 2 output."""

    id: int
    text: str
    timestamp_hint: Tuple[float, float]


@dataclass
class Scene:
    """One scene from source video preprocessing.

    ``embedding`` must be in the same space as the text encoder output and is
    expected to be L2-normalised. If not normalised, cosine similarity is still
    computed correctly here but downstream metrics may behave differently.
    """

    id: int
    start: float
    end: float
    embedding: np.ndarray


@dataclass
class Assignment:
    """One group of sentences assigned to a single scene or to generation."""

    sentence_ids: List[int]
    scene_id: int
    best_similarity: float          # weighted similarity (cosine * temporal_weight)
    raw_cosine: float               # raw cosine before temporal weighting
    temporal_weight: float          # the weight applied to the locked scene
    action: str                     # "retrieve" or "generate"
    timestamp_hint_merged: Tuple[float, float]
    # Per-step weighted similarity trail, kept for debugging and threshold tuning.
    similarity_trail: List[float] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _cosine(a: np.ndarray, b: np.ndarray) -> float:
    """Cosine similarity between two 1-D vectors. Safe against zero vectors."""

    na = float(np.linalg.norm(a))
    nb = float(np.linalg.norm(b))
    if na == 0.0 or nb == 0.0:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


def _cosine_to_all(query: np.ndarray, matrix: np.ndarray) -> np.ndarray:
    """Cosine similarity of ``query`` against every row of ``matrix``.

    ``matrix`` shape: ``(num_scenes, dim)``. Returns shape ``(num_scenes,)``.
    """

    q_norm = float(np.linalg.norm(query))
    if q_norm == 0.0:
        return np.zeros(matrix.shape[0], dtype=np.float32)
    row_norms = np.linalg.norm(matrix, axis=1)
    row_norms = np.where(row_norms == 0.0, 1.0, row_norms)
    return (matrix @ query) / (row_norms * q_norm)


def _stack_scene_embeddings(scenes: Sequence[Scene]) -> np.ndarray:
    return np.stack([s.embedding for s in scenes], axis=0)


def _scene_centers(scenes: Sequence[Scene]) -> np.ndarray:
    """Time midpoint of each scene in seconds."""

    return np.array([(s.start + s.end) / 2.0 for s in scenes], dtype=np.float32)


def _gaussian_temporal_weights(
    scene_centers: np.ndarray,
    hint_center: float,
    sigma: float,
) -> np.ndarray:
    """Gaussian weight in [0, 1] for every scene given a hint center time.

    A scene whose centre matches ``hint_center`` gets weight 1.0; scenes that
    are ``sigma`` seconds away get weight ~0.61; ``2*sigma`` away ~0.14;
    ``3*sigma`` away ~0.01. The weight never reaches zero, so distant scenes
    can still be selected if their visual similarity is overwhelmingly strong,
    but they are heavily suppressed.
    """

    if sigma <= 0.0:
        # Degenerate: no temporal prior. Return all-ones.
        return np.ones_like(scene_centers, dtype=np.float32)
    delta = scene_centers - float(hint_center)
    return np.exp(-(delta ** 2) / (2.0 * sigma * sigma)).astype(np.float32)


# ---------------------------------------------------------------------------
# Core: RetrievalGate
# ---------------------------------------------------------------------------


@dataclass
class RetrievalGateConfig:
    gate_threshold: float = 0.13       # tuned for SigLIP 2 raw cosine + temporal prior
    extend_epsilon: float = 0.03
    max_group_size: int = 5
    join_sep: str = " "
    temporal_sigma: float = 30.0       # seconds; controls Gaussian decay width
    enable_temporal_prior: bool = True


class RetrievalGate:
    """Greedy forward-walk grouping with retrieval/generation gating.

    Each candidate group is scored against every scene as
    ``weighted_sim = cosine(text_emb, scene_emb) * gaussian_weight(scene_time, hint_time)``
    where ``gaussian_weight`` decays with the distance between the scene's
    centre time and the centre of the merged ``source_timestamp_hint`` of the
    current group. The decision gate compares the final weighted similarity
    of the locked scene to ``gate_threshold``.

    Algorithm summary:
      i = 0
      while i < N:
          form a group starting at i, anchored to the locked scene S_locked
              S_locked = argmax_scene weighted_sim(encode(text_i), scene_emb,
                                                  hint_center_i)
          try to extend the group by sentence i+1, i+2, ...
              extension is accepted if and only if:
                  (a) the new best scene for the extended group is still
                      S_locked, and
                  (b) weighted similarity to S_locked did not drop by more than
                      extend_epsilon below the previous similarity
          on rejection: close the group
          decision gate on the final group weighted similarity:
              >= gate_threshold -> action = "retrieve"
              < gate_threshold  -> action = "generate"
          i = i + len(group)
    """

    def __init__(
        self,
        text_encoder: TextEncoder,
        config: Optional[RetrievalGateConfig] = None,
    ) -> None:
        self.encoder = text_encoder
        self.config = config or RetrievalGateConfig()

    def run(
        self,
        sentences: Sequence[Sentence],
        scenes: Sequence[Scene],
    ) -> List[Assignment]:
        if not sentences:
            return []
        if not scenes:
            raise ValueError("At least one scene is required.")

        scene_matrix = _stack_scene_embeddings(scenes)
        scene_centers = _scene_centers(scenes)
        n = len(sentences)
        assignments: List[Assignment] = []

        i = 0
        while i < n:
            assignment = self._build_group(
                i, sentences, scenes, scene_matrix, scene_centers
            )
            assignments.append(assignment)
            consumed = len(assignment.sentence_ids)
            # Defensive: must always advance to prevent infinite loops.
            if consumed < 1:
                raise RuntimeError(
                    f"Group at index {i} consumed zero sentences; refusing to loop."
                )
            i += consumed

        return assignments

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _hint_center(self, sentences: Sequence[Sentence], ids: Sequence[int]) -> float:
        """Centre of the merged timestamp hint across the group."""

        lo = sentences[ids[0]].timestamp_hint[0]
        hi = sentences[ids[-1]].timestamp_hint[1]
        return (float(lo) + float(hi)) / 2.0

    def _weighted_sims(
        self,
        text_emb: np.ndarray,
        scene_matrix: np.ndarray,
        scene_centers: np.ndarray,
        hint_center: float,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Return (weighted, raw_cosine, weights) for every scene."""

        raw = _cosine_to_all(text_emb, scene_matrix)
        if self.config.enable_temporal_prior:
            weights = _gaussian_temporal_weights(
                scene_centers, hint_center, self.config.temporal_sigma
            )
        else:
            weights = np.ones_like(raw, dtype=np.float32)
        weighted = raw * weights
        return weighted, raw, weights

    def _build_group(
        self,
        start: int,
        sentences: Sequence[Sentence],
        scenes: Sequence[Scene],
        scene_matrix: np.ndarray,
        scene_centers: np.ndarray,
    ) -> Assignment:
        cfg = self.config
        sep = cfg.join_sep

        # Seed: group of one sentence, lock to its best (weighted) scene.
        group_ids: List[int] = [start]
        joined_text = sentences[start].text
        joined_emb = self.encoder.encode(joined_text)

        hint_center = self._hint_center(sentences, group_ids)
        weighted, raw, weights = self._weighted_sims(
            joined_emb, scene_matrix, scene_centers, hint_center
        )
        locked_idx = int(np.argmax(weighted))
        best_weighted = float(weighted[locked_idx])
        best_raw = float(raw[locked_idx])
        best_weight = float(weights[locked_idx])
        sim_trail: List[float] = [best_weighted]

        # Try to extend.
        n = len(sentences)
        while (
            start + len(group_ids) < n
            and len(group_ids) < cfg.max_group_size
        ):
            next_idx = start + len(group_ids)
            candidate_text = joined_text + sep + sentences[next_idx].text
            candidate_emb = self.encoder.encode(candidate_text)

            candidate_ids = group_ids + [next_idx]
            candidate_hint_center = self._hint_center(sentences, candidate_ids)
            cand_weighted, cand_raw, cand_weights = self._weighted_sims(
                candidate_emb, scene_matrix, scene_centers, candidate_hint_center
            )
            candidate_best_idx = int(np.argmax(cand_weighted))
            candidate_locked_weighted = float(cand_weighted[locked_idx])

            # Extension accepted only if:
            #   - the candidate group still maps best to the locked scene
            #   - weighted similarity to the locked scene did not drop too far
            same_scene = candidate_best_idx == locked_idx
            tolerable_drop = (
                candidate_locked_weighted >= best_weighted - cfg.extend_epsilon
            )
            if not (same_scene and tolerable_drop):
                break

            group_ids.append(next_idx)
            joined_text = candidate_text
            joined_emb = candidate_emb
            best_weighted = candidate_locked_weighted
            best_raw = float(cand_raw[locked_idx])
            best_weight = float(cand_weights[locked_idx])
            sim_trail.append(best_weighted)

        # Decision gate is on the weighted similarity.
        action = "retrieve" if best_weighted >= cfg.gate_threshold else "generate"

        hint_start = sentences[group_ids[0]].timestamp_hint[0]
        hint_end = sentences[group_ids[-1]].timestamp_hint[1]

        return Assignment(
            sentence_ids=group_ids,
            scene_id=scenes[locked_idx].id,
            best_similarity=best_weighted,
            raw_cosine=best_raw,
            temporal_weight=best_weight,
            action=action,
            timestamp_hint_merged=(float(hint_start), float(hint_end)),
            similarity_trail=sim_trail,
        )


# ---------------------------------------------------------------------------
# FrameSelector (used by Phase 5 for generation conditioning)
# ---------------------------------------------------------------------------


@dataclass
class FrameSelectorConfig:
    # Strategy for picking a representative frame from the locked scene.
    # "middle"      -> the frame nearest the midpoint of the merged hint range
    # "best_clip"   -> the frame with the highest CLIP/SigLIP similarity to the
    #                  joined sentence text (requires per-frame embeddings)
    strategy: str = "middle"


class FrameSelector:
    """Pick a representative frame from the locked scene for Phase 5.

    Two strategies are supported. ``"middle"`` is dependency-free and always
    available. ``"best_clip"`` requires per-frame embeddings and a text encoder;
    if either is missing, the selector falls back to ``"middle"``.

    The frame chosen here becomes the image-conditioning input for the
    image-to-video diffusion model in Phase 5.
    """

    def __init__(
        self,
        config: Optional[FrameSelectorConfig] = None,
        text_encoder: Optional[TextEncoder] = None,
    ) -> None:
        self.config = config or FrameSelectorConfig()
        self.encoder = text_encoder

    def select(
        self,
        assignment: Assignment,
        scene: Scene,
        frames: Sequence["FrameRef"],
        joined_sentence_text: Optional[str] = None,
    ) -> "FrameRef":
        """Return the chosen frame.

        ``frames`` is the sequence of available frames inside ``scene``, each
        carrying its timestamp and an optional embedding. The selector narrows
        to frames inside ``assignment.timestamp_hint_merged`` first; if the
        narrowed window is empty (hints can lie outside the scene if Phase 2
        timestamps drift), it falls back to all scene frames.
        """

        if not frames:
            raise ValueError(f"Scene {scene.id} has no frames available.")

        lo, hi = assignment.timestamp_hint_merged
        in_window = [f for f in frames if lo <= f.timestamp <= hi]
        candidates = in_window if in_window else list(frames)

        strategy = self.config.strategy
        if strategy == "best_clip":
            if (
                self.encoder is not None
                and joined_sentence_text is not None
                and all(f.embedding is not None for f in candidates)
            ):
                text_emb = self.encoder.encode(joined_sentence_text)
                frame_matrix = np.stack(
                    [f.embedding for f in candidates], axis=0  # type: ignore[arg-type]
                )
                sims = _cosine_to_all(text_emb, frame_matrix)
                return candidates[int(np.argmax(sims))]
            # Fall through to middle if dependencies missing.

        # Default / fallback: middle of window (or middle of all frames).
        midpoint = (candidates[0].timestamp + candidates[-1].timestamp) / 2.0
        return min(candidates, key=lambda f: abs(f.timestamp - midpoint))


@dataclass
class FrameRef:
    """Reference to one frame, optionally with an embedding."""

    timestamp: float
    path: str  # path on disk or whatever the rest of the pipeline expects
    embedding: Optional[np.ndarray] = None


# ---------------------------------------------------------------------------
# Convenience: run end-to-end and summarise
# ---------------------------------------------------------------------------


def summarise_assignments(assignments: Sequence[Assignment]) -> dict:
    """Quick stats useful for smoke tests and threshold tuning."""

    if not assignments:
        return {"num_groups": 0}

    group_sizes = [len(a.sentence_ids) for a in assignments]
    weighted_sims = [a.best_similarity for a in assignments]
    raw_sims = [a.raw_cosine for a in assignments]
    weights = [a.temporal_weight for a in assignments]
    actions = [a.action for a in assignments]

    return {
        "num_groups": len(assignments),
        "num_sentences": sum(group_sizes),
        "group_size_min": min(group_sizes),
        "group_size_max": max(group_sizes),
        "group_size_mean": sum(group_sizes) / len(group_sizes),
        "num_singletons": sum(1 for s in group_sizes if s == 1),
        "num_multi": sum(1 for s in group_sizes if s > 1),
        "weighted_sim_min": min(weighted_sims),
        "weighted_sim_max": max(weighted_sims),
        "weighted_sim_mean": sum(weighted_sims) / len(weighted_sims),
        "raw_cosine_min": min(raw_sims),
        "raw_cosine_max": max(raw_sims),
        "raw_cosine_mean": sum(raw_sims) / len(raw_sims),
        "temporal_weight_min": min(weights),
        "temporal_weight_max": max(weights),
        "temporal_weight_mean": sum(weights) / len(weights),
        "num_retrieve": sum(1 for a in actions if a == "retrieve"),
        "num_generate": sum(1 for a in actions if a == "generate"),
    }
```

### Definition of done

- File matches the code above byte-for-byte.
- `python -c "from src.phase4_retrieve import RetrievalGate, RetrievalGateConfig"` succeeds.

---

## Task 2: Update `configs/default.yaml`

In the `phase4:` block, update the existing keys and add the two new ones. The block should now read:

```yaml
phase4:
  gate_threshold: 0.13         # lowered from 0.25 to fit SigLIP 2 raw cosine range with temporal weighting
  extend_epsilon: 0.03
  max_group_size: 5
  join_sep: " "
  temporal_sigma: 30.0         # seconds; Gaussian decay width for temporal prior
  enable_temporal_prior: true  # set to false to disable for ablation
```

In `src/pipeline.py`, update the `RetrievalGateConfig(...)` instantiation to pass the two new keys:

```python
config=RetrievalGateConfig(
    gate_threshold=cfg["phase4"]["gate_threshold"],
    extend_epsilon=cfg["phase4"]["extend_epsilon"],
    max_group_size=cfg["phase4"]["max_group_size"],
    join_sep=cfg["phase4"]["join_sep"],
    temporal_sigma=cfg["phase4"]["temporal_sigma"],
    enable_temporal_prior=cfg["phase4"]["enable_temporal_prior"],
),
```

### Definition of done

- YAML parses cleanly: `python -c "import yaml; yaml.safe_load(open('configs/default.yaml'))"`.
- Pipeline imports cleanly: `python -c "import src.pipeline"`.

---

## Task 3: Update the smoke-test print block in `src/pipeline.py`

Find the existing smoke-test print block introduced in the previous Phase 4 task. Update it to surface the new fields (`raw_cosine` and `temporal_weight`). Replace it with:

```python
from src.phase4_retrieve import summarise_assignments
print("=== PHASE 4 SMOKE TEST ===")
print("Assignments:")
for a in assignments:
    print(
        f"  group sents={a.sentence_ids} "
        f"scene={a.scene_id} "
        f"weighted={a.best_similarity:.3f} "
        f"raw={a.raw_cosine:.3f} "
        f"weight={a.temporal_weight:.3f} "
        f"action={a.action} "
        f"hint={a.timestamp_hint_merged} "
        f"trail={[round(x, 3) for x in a.similarity_trail]}"
    )
print("Summary:", summarise_assignments(assignments))
print("=== END PHASE 4 SMOKE TEST ===")
```

### Definition of done

- Print block updated; the next run on `review_2.mp4` shows `weighted=`, `raw=`, and `weight=` per assignment.

---

## Task 4: Smoke test on `review_2.mp4`

Run the pipeline end-to-end on `data/eval_videos/review_2.mp4` and capture the output.

Report all assignments and the summary stats. Compare to the previous run on the same video.

### Report format

```
=== TEMPORAL PRIOR UPDATE REPORT ===

Code update: yes / no
Config update: yes / no
Smoke-test print block updated: yes / no

Pipeline run on review_2.mp4: success / failure
If failure, traceback: <pasted>

Assignments (from new run):
  <full output of print block>

Summary stats (from new run):
  <full summarise_assignments dict>

Comparison to previous run:
  Sentence 0 was: scene=80 sim=0.145 / now: scene=<X> weighted=<Y> raw=<Z> weight=<W>
  Sentence 1 was: scene=18 sim=0.106 / now: scene=<X> weighted=<Y> raw=<Z> weight=<W>
  Sentence 2 was: scene=40 sim=0.171 / now: scene=<X> weighted=<Y> raw=<Z> weight=<W>
  Sentences [3,4] was: scene=24 sim=0.109 / now: scene=<X> weighted=<Y> raw=<Z> weight=<W>

Action distribution:
  retrieve: <n>
  generate: <n>

Observation: <one sentence on whether scene choices look more temporally sensible than the previous run>

Blockers: <list, or "none">
```

## Hard constraints

- Do not touch `src/phase2_summarize.py` or Phase 2 logic.
- Do not touch Phase 5 generation or pipeline assembly.
- Do not modify the algorithm in `src/phase4_retrieve.py` beyond pasting the new content verbatim.
- Do not commit or push.

End of brief.
