# Phase 4 Refactor — Implementation Brief for Gemini Agent

## Role and rules of engagement

You are executing a focused refactor of Phase 4 of a video summarization pipeline. The new algorithm and its core code have already been designed and written by Claude (the user's planning assistant). Your job is execution: apply the changes, wire them in, run the smoke test, and report back.

**Do not redesign the algorithm.** The code in Task 2 below is the authoritative implementation. Use it verbatim.

**Do not start unrelated work.** This brief is the only scope. If you find issues outside Phase 4, log them at the end of your report but do not act on them.

**Behavior expectations:**

- Be direct. No filler.
- If a task has a blocker (file does not exist, function signature does not match, dependency missing), stop that task, log the blocker with exact paths and error messages, and continue with tasks that are not blocked.
- Do not invent file paths. If a path in this brief does not exist in the repo, search for the closest match and report what you found.
- All file edits must preserve existing behavior outside the scope listed.
- Backup before delete. Never hard-delete a file; move to `archive/` instead.
- Use Python 3.10+ syntax (the existing pipeline uses it).
- After every task, print `=== TASK N DONE ===` to stdout so progress is easy to track.

## Project context

The pipeline takes long-form narrated videos (e.g. product reviews, tutorials), transcribes them, summarizes the narration with a local LLM, synthesizes voice-over, then assembles a shorter summary video by matching narration sentences to clips. The thesis pivot in this session changed Phase 4 from a constrained sequence alignment problem (CCMA / DP / Hungarian) to a simpler grouping-based retrieval with a decision gate that routes weak matches to a future Phase 5 (image-to-video generation).

The new Phase 4 does three things:

1. **Group consecutive sentences** that map best to the same scene in the source video.
2. **Retrieve** that scene when similarity is above a threshold.
3. **Mark for generation** when similarity is below the threshold (Phase 5 will handle generation later; for now, log a warning and fall back to retrieve).

## Prerequisite state (already done)

A Phase 2 prompt update has already been applied in a previous task. As a result:

- The `SummaryScript` Pydantic model in `src/schemas.py` no longer has a `style` field.
- The Phase 2 summarization function (`Phase2Summarizer.run` in `src/phase2_summarize.py`) no longer accepts a `style` parameter.
- The Phase 2 output JSON has the structure:
  ```
  {
    "target_duration": <int>,
    "sentences": [
      {
        "id": <int>,
        "text": "<narration sentence>",
        "estimated_duration_seconds": <float>,
        "source_timestamp_hint": [<float>, <float>],
        "keywords": ["<concrete visual>", ...]
      },
      ...
    ]
  }
  ```
- Sentences may exhibit topical grouping (consecutive sentences elaborating the same subject). This is intentional and is what the new Phase 4 in this brief is designed to consume.

When wiring Phase 4 in Task 4, use this structure. Do NOT expect a `style` field anywhere.

## Task overview

| Task | Description | Blocking next? |
|---|---|---|
| 1 | Archive obsolete files (CCMA, DP, MOTA, Hungarian) | No |
| 2 | Replace `src/phase4_retrieve.py` with new code (hard-coded below) | Yes, blocks 4 and 5 |
| 3 | Update `configs/default.yaml` with new Phase 4 config | Yes, blocks 4 |
| 4 | Wire the new Phase 4 into `src/pipeline.py` | Yes, blocks 5 |
| 5 | Run smoke test on one video, print stats | Yes, blocks 6 |
| 6 | Regression check (import errors, existing tests) | No |

Execute in order 1 → 2 → 3 → 4 → 5 → 6.

---

## Task 1: Archive obsolete files

### Goal

Move all code related to the old alignment approach out of the active codebase. Do not delete; move to `archive/` so it can be referenced later if needed.

### Steps

1. Create directory `archive/` at repo root if it does not exist.
2. Move (not copy) these files into `archive/`, preserving their basenames:
   - `ccma_fix_prompt.md` (if it exists at repo root)
   - `cv_align_implementation_prompt.md` (if it exists at repo root)
3. Inside `src/phase4_retrieve.py`, identify and **list** (do not delete yet) all functions, classes, and module-level code related to:
   - CCMA (anything with `ccma` in the name, e.g. `ccma_align_sequence`)
   - DP / Viterbi alignment (anything with `dp_align`, `viterbi`, etc.)
   - Hungarian assignment (anything with `hungarian` in the name)
   - MOTA metrics (anything with `mota` in the name)
4. Before Task 2 replaces this file, copy the current `src/phase4_retrieve.py` to `archive/phase4_old.py` so the old code is preserved.
5. In `src/utils/metrics.py` (if it exists), identify but DO NOT remove anything related to MOTA. Just list it in your report. SigLIP similarity functions must stay.

### Definition of done

- `archive/phase4_old.py` exists and contains the previous Phase 4 code.
- `archive/` contains `ccma_fix_prompt.md` and `cv_align_implementation_prompt.md` if they existed.
- Report lists every old function name and its line number in the original file.

### If blocked

- File `src/phase4_retrieve.py` does not exist → search for closest match (e.g. `src/retrieval.py`, `src/phase4.py`). Use the closest match path for Task 2 as well. Report the path you used.

---

## Task 2: Replace `src/phase4_retrieve.py`

### Goal

Replace the existing Phase 4 file with the new grouping-based retrieval implementation.

### Steps

1. Confirm Task 1 step 4 was done (old file backed up to `archive/phase4_old.py`).
2. Overwrite `src/phase4_retrieve.py` with the exact content in the code block below. No edits, no formatting changes.

### Code (use verbatim)

```python
"""Phase 4: Grouping-based retrieval with decision gate.

Replaces the prior CCMA / DP / Hungarian alignment approach. The new design:

1. Walk the narration sentences forward and greedily form groups of consecutive
   sentences that all map best to the same scene in the source video.
2. For each group, decide whether the best matching scene is similar enough to
   be retrieved as-is, or whether the similarity is too weak and the group
   should be routed to Phase 5 (image-to-video generation).

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
    best_similarity: float
    action: str  # "retrieve" or "generate"
    timestamp_hint_merged: Tuple[float, float]
    # Per-step similarity trail, kept for debugging and threshold tuning.
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


# ---------------------------------------------------------------------------
# Core: RetrievalGate
# ---------------------------------------------------------------------------


@dataclass
class RetrievalGateConfig:
    gate_threshold: float = 0.25
    extend_epsilon: float = 0.03
    max_group_size: int = 5
    join_sep: str = " "


class RetrievalGate:
    """Greedy forward-walk grouping with retrieval/generation gating.

    Algorithm summary:
      i = 0
      while i < N:
          form a group starting at i, anchored to scene S_locked
              (S_locked = argmax_scene cosine(encode(text_i), scene_emb))
          try to extend the group by sentence i+1, i+2, ...
              extension is accepted if and only if:
                  (a) the new best scene for the extended group is still
                      S_locked, and
                  (b) similarity to S_locked did not drop by more than
                      extend_epsilon below the previous similarity
          on rejection: close the group
          decision gate on the final group similarity:
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
        n = len(sentences)
        assignments: List[Assignment] = []

        i = 0
        while i < n:
            assignment = self._build_group(i, sentences, scenes, scene_matrix)
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

    def _build_group(
        self,
        start: int,
        sentences: Sequence[Sentence],
        scenes: Sequence[Scene],
        scene_matrix: np.ndarray,
    ) -> Assignment:
        cfg = self.config
        sep = cfg.join_sep

        # Seed: group of one sentence, lock to its best scene.
        group_ids: List[int] = [start]
        joined_text = sentences[start].text
        joined_emb = self.encoder.encode(joined_text)

        sims = _cosine_to_all(joined_emb, scene_matrix)
        locked_idx = int(np.argmax(sims))
        best_sim = float(sims[locked_idx])
        sim_trail: List[float] = [best_sim]

        # Try to extend.
        n = len(sentences)
        while (
            start + len(group_ids) < n
            and len(group_ids) < cfg.max_group_size
        ):
            next_idx = start + len(group_ids)
            candidate_text = joined_text + sep + sentences[next_idx].text
            candidate_emb = self.encoder.encode(candidate_text)
            candidate_sims = _cosine_to_all(candidate_emb, scene_matrix)
            candidate_best_idx = int(np.argmax(candidate_sims))
            candidate_locked_sim = float(candidate_sims[locked_idx])

            # Extension accepted only if:
            #   - the candidate group still maps best to the locked scene
            #   - similarity to the locked scene did not drop too far
            same_scene = candidate_best_idx == locked_idx
            tolerable_drop = candidate_locked_sim >= best_sim - cfg.extend_epsilon
            if not (same_scene and tolerable_drop):
                break

            group_ids.append(next_idx)
            joined_text = candidate_text
            joined_emb = candidate_emb
            best_sim = candidate_locked_sim
            sim_trail.append(best_sim)

        # Decision gate.
        action = "retrieve" if best_sim >= cfg.gate_threshold else "generate"

        hint_start = sentences[group_ids[0]].timestamp_hint[0]
        hint_end = sentences[group_ids[-1]].timestamp_hint[1]

        return Assignment(
            sentence_ids=group_ids,
            scene_id=scenes[locked_idx].id,
            best_similarity=best_sim,
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
    sims = [a.best_similarity for a in assignments]
    actions = [a.action for a in assignments]

    return {
        "num_groups": len(assignments),
        "num_sentences": sum(group_sizes),
        "group_size_min": min(group_sizes),
        "group_size_max": max(group_sizes),
        "group_size_mean": sum(group_sizes) / len(group_sizes),
        "num_singletons": sum(1 for s in group_sizes if s == 1),
        "num_multi": sum(1 for s in group_sizes if s > 1),
        "similarity_min": min(sims),
        "similarity_max": max(sims),
        "similarity_mean": sum(sims) / len(sims),
        "num_retrieve": sum(1 for a in actions if a == "retrieve"),
        "num_generate": sum(1 for a in actions if a == "generate"),
    }
```

### Definition of done

- `src/phase4_retrieve.py` content matches the code block above byte-for-byte.
- `python -c "from src.phase4_retrieve import RetrievalGate, RetrievalGateConfig, Sentence, Scene, Assignment, FrameSelector, FrameRef, summarise_assignments"` succeeds with no error.

### If blocked

- Import path differs (e.g. project uses flat layout, no `src/` prefix) → use the layout the rest of the project uses. Report which path you used.

---

## Task 3: Update `configs/default.yaml`

### Goal

Add new Phase 4 config, remove old CCMA / DP / MOTA / Hungarian config keys.

### Steps

1. Open `configs/default.yaml`.
2. **Remove** any top-level or nested keys whose names contain any of: `ccma`, `dp_align`, `viterbi`, `hungarian`, `mota`. Comment out, do not delete, in case a key is shared with another subsystem you do not recognise. Add an inline comment `# removed in phase 4 refactor` next to each comment-out.
3. **Add** the following block at the end of the file (or under an existing `phase4:` block if one exists — in which case merge keys):

```yaml
phase4:
  gate_threshold: 0.25       # below this similarity, group routes to phase 5 generation
  extend_epsilon: 0.03       # max similarity drop tolerated when extending a group
  max_group_size: 5          # safety cap on group size
  join_sep: " "              # separator used when concatenating sentences for re-embedding
```

### Definition of done

- New `phase4:` block exists with all four keys above.
- All old keys matching the patterns above are commented out with the marker comment.
- YAML still parses: `python -c "import yaml; yaml.safe_load(open('configs/default.yaml'))"` succeeds.

### If blocked

- File does not exist → search for `*.yaml` under `configs/` and use the one referenced by `src/pipeline.py`. Report the path used.

---

## Task 4: Wire new Phase 4 into `src/pipeline.py`

### Goal

Replace the old Phase 4 call with the new `RetrievalGate.run()` call. For `action == "generate"` assignments, log a warning and fall back to retrieve (Phase 5 is not yet implemented).

### Steps

1. Open `src/pipeline.py`. Locate the Phase 4 invocation. Likely candidates:
   - Function or method named `phase4_*`, `retrieve_*`, `align_*`
   - Import lines `from src.phase4_retrieve import ...` or similar
2. Identify two pieces of context the old code consumed:
   - The list of summary sentences from Phase 2 output (with text and timestamp hints)
   - The list of scenes from preprocessing (with start, end, embedding)
3. Identify the text encoder used to embed scenes. It is most likely a SigLIP wrapper, exposed somewhere as `text_encoder`, `siglip_encoder`, or via a factory like `load_text_encoder()`. If it only has an image encoder, find or instantiate the matching text tower from the same model id.
4. Replace the old Phase 4 call with:

```python
from src.phase4_retrieve import (
    RetrievalGate,
    RetrievalGateConfig,
    Sentence as P4Sentence,
    Scene as P4Scene,
)
import logging

logger = logging.getLogger(__name__)

# Adapt Phase 2 sentences to P4Sentence
p4_sentences = [
    P4Sentence(
        id=s["id"],
        text=s["text"],
        timestamp_hint=tuple(s["source_timestamp_hint"]),
    )
    for s in phase2_output["sentences"]
]

# Adapt scenes to P4Scene
p4_scenes = [
    P4Scene(
        id=sc.id,                 # adapt to actual scene attribute name
        start=sc.start,           # adapt
        end=sc.end,               # adapt
        embedding=sc.embedding,   # adapt; must be np.ndarray shape (D,)
    )
    for sc in scenes
]

# Run gate
gate = RetrievalGate(
    text_encoder=text_encoder,
    config=RetrievalGateConfig(
        gate_threshold=cfg["phase4"]["gate_threshold"],
        extend_epsilon=cfg["phase4"]["extend_epsilon"],
        max_group_size=cfg["phase4"]["max_group_size"],
        join_sep=cfg["phase4"]["join_sep"],
    ),
)
assignments = gate.run(p4_sentences, p4_scenes)

# Phase 5 fallback: for now, "generate" actions fall back to retrieve with a warning.
for a in assignments:
    if a.action == "generate":
        logger.warning(
            "Phase 5 not implemented yet; falling back to retrieve for group %s "
            "(scene %d, sim=%.3f).",
            a.sentence_ids, a.scene_id, a.best_similarity,
        )

# Downstream (Phase 6 assembly) consumes `assignments`.
```

5. Update the downstream assembly call (Phase 6 or equivalent) to consume `assignments` instead of whatever the old Phase 4 returned. The minimal contract is:
   - For each `Assignment`, retrieve the source video clip corresponding to `scene_id` (using `scene.start`, `scene.end`).
   - Pair that clip with the audio for sentences `sentence_ids` (which may be more than one).
   - Clip duration may need to be stretched or trimmed to fit the audio duration of the joined sentences. **If existing assembly already does this for the old output, leave it alone.**

6. **Do not implement Phase 5 generation.** Just the warning is enough for now.

### Definition of done

- Pipeline imports the new module and calls `RetrievalGate.run()`.
- Pipeline no longer imports anything from the old Phase 4 (CCMA, DP, Hungarian, MOTA).
- Pipeline runs end-to-end on one video without crashing (verified in Task 5).
- Warning is logged for at least zero `generate` assignments (warning code path executes only when needed).

### If blocked

- Cannot locate `text_encoder` instance → search for the place where scene embeddings are computed during preprocessing. The same encoder must be reused. Report the path.
- Scene dataclass uses different field names → adapt the comprehension. Report what attribute names you mapped.
- Phase 2 output structure differs from the example (e.g. uses a dataclass, not a dict) → adapt the comprehension. Report the structure.

---

## Task 5: Smoke test on one video

### Goal

Confirm the new Phase 4 produces sensible output on a real video. Print stats for review.

### Steps

1. Pick one video from the dataset. Use whatever entry point the project provides (e.g. `python -m src.pipeline --video <id>`).
2. Add a temporary `print()` block right after the `gate.run(...)` call:

```python
from src.phase4_retrieve import summarise_assignments
print("=== PHASE 4 SMOKE TEST ===")
print("Assignments:")
for a in assignments:
    print(
        f"  group sents={a.sentence_ids} "
        f"scene={a.scene_id} "
        f"sim={a.best_similarity:.3f} "
        f"action={a.action} "
        f"hint={a.timestamp_hint_merged} "
        f"trail={[round(x, 3) for x in a.similarity_trail]}"
    )
print("Summary:", summarise_assignments(assignments))
print("=== END PHASE 4 SMOKE TEST ===")
```

3. Run the pipeline on the chosen video. Capture stdout.
4. Inspect the output and verify all of the following:
   - `summary["num_groups"] > 0`
   - `summary["num_sentences"]` equals the total sentence count from Phase 2
   - `summary["similarity_min"] >= 0.0` and `similarity_max <= 1.0`
   - The list of assignments covers every sentence id from 0 to N-1 exactly once, in order, with no gaps and no overlaps

5. Note any of these flags (these are observations to report, not errors):
   - `num_multi == 0` → grouping behavior did not emerge, possibly because the LLM output is too topic-diverse or `extend_epsilon` is too strict.
   - `num_retrieve == 0` or `num_generate == 0` → threshold may be off.
   - `similarity_mean < 0.10` or `> 0.80` → embedding-space scale is unusual, worth flagging.

6. Remove the temporary print block, or leave it behind a debug flag. Report which you did.

### Definition of done

- Stdout from the run is captured and included in the report.
- All four coverage checks pass.
- Observation flags are noted in the report.

### If blocked

- Pipeline crashes inside Phase 4 → capture the full traceback and report. Do not attempt to fix; the algorithm is fixed and the bug is almost certainly in the wiring (Task 4).
- No scene embeddings found → preprocessing has not been run. Run preprocessing first if the project provides a command; otherwise report.

---

## Task 6: Regression check

### Goal

Confirm no other part of the pipeline broke.

### Steps

1. Run `pytest` (or `pytest -q`) from repo root if a `tests/` directory exists. Capture pass/fail counts.
2. Grep the repo for any remaining imports of removed symbols. Run:

```bash
grep -rn "ccma\|dp_align\|viterbi\|hungarian\|mota" --include="*.py" src/ scripts/ tests/ 2>/dev/null || true
```

Each hit must be reviewed. Hits in `archive/` are acceptable. Hits anywhere else are regressions; report file:line and context, do not silently delete.

3. Run `python -c "import src.pipeline"` to confirm no import-time errors.

### Definition of done

- Test count and failure list reported.
- Grep output reported, with file:line for each non-archive hit.
- `import src.pipeline` succeeds with no error.

### If blocked

- No `tests/` directory → skip step 1, note it in the report.

---

## Final report format

Append a single block at the end of your work containing:

```
=== FINAL REPORT ===

Task 1 (archive):
  - Moved files: <list with paths>
  - Old Phase 4 functions identified: <list with line numbers>
  - MOTA references in metrics.py: <list>

Task 2 (replace phase4_retrieve.py):
  - Path used: <path>
  - Import smoke test: <pass/fail>

Task 3 (config):
  - Old keys commented out: <list>
  - New phase4 block added: yes/no
  - YAML parse check: <pass/fail>

Task 4 (pipeline wire-up):
  - text_encoder path used: <where it came from>
  - Scene attribute mapping: id=<>, start=<>, end=<>, embedding=<>
  - Phase 2 output structure: <dict/dataclass/etc>

Task 5 (smoke test):
  - Video used: <id or path>
  - num_groups: <n>
  - num_sentences: <n>
  - num_singletons / num_multi: <n> / <n>
  - num_retrieve / num_generate: <n> / <n>
  - similarity range: <min>..<max> (mean=<mean>)
  - Observation flags hit: <list>
  - Full assignments output: <pasted>

Task 6 (regression):
  - Test results: <pass>/<fail>
  - Non-archive grep hits: <list>
  - Pipeline import: <pass/fail>

Blockers encountered: <list, or "none">
Out-of-scope issues noticed: <list, or "none">

=== END REPORT ===
```

## Hard constraints (do not violate)

- Do not modify `src/phase4_retrieve.py` content beyond what Task 2 specifies.
- Do not implement Phase 5 generation. Just the warning.
- Do not change `configs/default.yaml` keys outside the scope of Task 3.
- Do not touch Phase 1 (Whisper), Phase 2 (LLM summarization), Phase 3 (TTS), or Phase 6 assembly logic except for the minimal Phase 4 → Phase 6 hand-off in Task 4.
- Do not commit or push. Leave changes uncommitted in the working tree.

End of brief.
