# Project Context Briefing Request

## Goal

I am a new collaborator joining mid-project. I have a handoff doc covering the pipeline architecture, current phase status, and recent technical decisions (Phase 4 locked, Phase 5 smoke tests in progress). What I lack is the **broader project context** — the academic framing, baselines, evaluation methodology, and history of design decisions.

Please produce a context briefing covering the sections below. Be factual and specific. If something is not knowable from the codebase, prior chat logs, or project artifacts, write "UNKNOWN — needs user input" rather than guessing.

## Sections to cover

### 1. Problem definition

- What is "narrated video summarization" as defined in this project? Concrete one-paragraph definition.
- How does it differ from standard video summarization (e.g., keyframe selection, video skimming)? What makes "narrated" distinct?
- What is the target input (video type, length, domain) and target output (length, format, modality)?

### 2. Three contributions, expanded

The handoff lists three contributions:
1. First application of hybrid retrieval-generation for narrated video summarization
2. Frame-grounded generation (I2V conditioned on retrieved keyframes)
3. Training-free consumer GPU pipeline

For each, explain:
- The precise claim (what exactly is "first"? what makes the pipeline "frame-grounded" vs. alternatives?).
- The evidence or argument needed to support it.
- Any prior work that comes close, and how this differs.

### 3. Baselines and comparisons

- What baselines, if any, is this thesis compared against? (Other summarization systems? Ablations? Human reference summaries?)
- If no formal baseline, what is the evaluation strategy?
- Are there reference datasets being benchmarked on (e.g., TVSum, SumMe, QFVS), or is this purely on the 10 in-house videos?

### 4. Evaluation methodology

- What are the final thesis-level metrics (as opposed to per-phase engineering metrics)?
- Are these automatic (e.g., ROUGE on transcripts, CLIP score on outputs), human evaluation, or both?
- What does "success" look like quantitatively?

### 5. Design history

- Why was the previous approach (CCMA / DP alignment) dropped? What specifically did not work?
- What were the alternatives considered before settling on hybrid retrieval-generation? Why was this chosen?
- Are there other major pivots in the project history worth knowing?

### 6. Academic context

- Program / degree level (undergrad, masters, PhD)?
- Institution and advisor (high-level — research group focus)?
- Expected thesis scope (page count, deliverable format, defense timeline)?
- Any constraints or commitments already made to the advisor that lock in design decisions?

### 7. Risk register

- What are the top 3 risks to thesis completion as currently understood?
- Which design decisions are reversible vs. locked-in?
- Is there a fallback if Phase 5 (I2V generation) cannot produce usable output?

## Format

Markdown, ~2 pages. Tables OK where useful. Cite specific files, commits, or chat logs where claims are grounded in artifacts. For anything that requires user input to answer, list it cleanly at the end under "Questions for user" so we can resolve them in one batch.

## Constraints

- Do not invent academic context (program, advisor, dataset choices) if not documented somewhere accessible.
- Do not paraphrase the existing handoff doc — assume the reader has it. Add only new information.
- If two sources conflict (e.g., codebase says X, old chat log says Y), flag the conflict.

=== END OF BRIEF ===
