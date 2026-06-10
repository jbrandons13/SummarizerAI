# FACET — Implementation & Experiment Brief

**Audience:** GPU coding agent (Gemini) with read/write access to the existing summarization pipeline.
**Hardware:** single RTX 3090, 24 GB. One model resident at a time.
**Objective:** Implement and evaluate (1) **Alt-1: probe-then-commit** (cheap DACA-equivalent selection) and (2) **FACET** (centroid concept embedding + block/timestep-targeted IP-Adapter injection + norm-share controller), against the existing **DACA** baseline, inside Phase 4 only. Produce frontier plots, a gate table, and a go/no-go verdict per component.

This brief is staged. Each stage has a **verification gate**. Do not start a stage before the previous gate passes (or is explicitly waived by the human). At checkpoints **G0–G4**, stop, write the report, and wait for human approval before continuing.

---

## 0. Ground rules (read before touching anything)

1. **Scope lock.** Only Phase 4 (per-shot image generation and its scale-selection logic) may be modified. Treat all Phase 1–3 code and caches as **read-only**.
2. **Reuse caches.** Transcript, summary/storyboard JSON, TTS audio, and the canonical concept reference images come from the existing cache of previous runs. Never recompute them. Assert their existence at startup; record file hashes in the run log.
3. **Phase 5 (image-to-video)** is run exactly once, at the very end (Stage 6), for the chosen ship config plus the DACA baseline on one video, for qualitative side-by-side. No I2V during experiments.
4. **One model resident at a time.** Generation passes load only the SDXL stack (UNet + VAE + text encoders + LoRA + IP-Adapter image encoder). Scoring passes load only DINOv2 (+ CLIP for CLIP-T). Between passes: `del pipe; gc.collect(); torch.cuda.empty_cache()`, then assert free VRAM > 20 GB. Exception: TAESD (`madebyollin/taesdxl`, ~5 M params) may co-reside with SDXL during Alt-1 probes.
5. **Determinism.** fp16, PyTorch SDPA (or xformers if already used in the repo — keep whatever the baseline used, do not change). **Deterministic scheduler only** (DDIM or Euler — whatever the existing DACA config uses, as long as it is non-ancestral; if the repo uses an ancestral scheduler, flag this at G0 and switch all experimental arms, including the DACA reproduction, to Euler so the comparison stays internally consistent). Per-shot fixed seeds (`seeds.json`), and **cached initial latents** passed explicitly via `latents=` so every arm of every shot starts from bitwise-identical noise.
6. **Config-driven.** One YAML (`configs/facet.yaml`) holds every knob in Appendix A. All runs write to `runs/<UTC-stamp>/`. Append-only JSONL logging (§6). Never overwrite a previous run.
7. **Do not change** image resolution, step count, CFG scale, negative prompt, LoRA weights/fusion state, or the W grid — read them from the existing DACA config and reuse verbatim. The point is a controlled comparison.

---

## 1. Scope & inputs (what is fixed, what is new)

| Phase | Status | Action |
|---|---|---|
| 1. Transcript | cached | load only; hash-check |
| 2. Summary + storyboard JSON | cached | load only; this defines shots, per-shot prompts, narration lines, concept ids |
| 3. TTS audio | cached | untouched until Stage 6 |
| Concept canonical reference (1 image/concept) | cached | load only; it remains the **evaluation** reference for ref-similarity in *all* arms, even when the *injected* embedding is a centroid |
| 4. Shot image generation | **all work happens here** | new module `pipeline/facet/` |
| 5. I2V render | cached code | run once in Stage 6 for ship config + DACA |

**One new cached artifact is allowed** (reference-side, one-time, cheap): K−1 = 5 extra reference *variants* per concept — same concept prompt, same LoRA, different fixed seeds — stored as `refs/variants/<concept_id>/var_{k}.png` with their embeddings cached as `.pt`. The canonical image itself is **not** regenerated.

During recon (§2) you must locate and record: the storyboard schema (field names for shot prompt, narration text used by CLIP-T, concept id), the DACA config (W grid, τ, scheduler, steps, CFG, resolution), the existing DINOv2/CLIP-T scoring entry points, and the known **near-reference control shot** in the ecology video (it must be flagged in config — see §6).

---

## 2. Stage 0 — Recon, harness, and DACA reproduction (gate G0)

### 2.1 Recon checklist
1. Print and record: `diffusers`/`torch` versions, pipeline class, scheduler class, steps, CFG, resolution, LoRA name + scale, IP-Adapter weight file.
2. Determine the IP-Adapter variant: inspect `pipe.unet.encoder_hid_proj.image_projection_layers[0]` — `ImageProjection` (4 tokens, global CLIP embed) = **vanilla**; `Resampler`-style (16 tokens, patch features) = **Plus**. This decides §5.1.
3. Dump `pipe.unet.attn_processors.keys()` to `unet_attn_map.txt`. Identify all cross-attention processors (names ending `attn2.processor`) grouped by block: `down_blocks.2.attentions.{0,1}`, `mid_block.attentions.0`, `up_blocks.0.attentions.{0,1,2}`, etc. These name prefixes are how layer gates are addressed later.
4. Check whether `pipe.set_ip_adapter_scale` accepts the per-layer dict form (InstantStyle-style). We will not rely on it at runtime (custom processors handle gating), but it is a useful cross-check during the block probe.
5. **Wrap, do not reimplement, the existing scoring code.** `scoring_wrap.py` must call the repo's own DINOv2 and CLIP-T functions with identical preprocessing, otherwise thesis numbers will not reproduce. Add on top: `c_s`, pairwise consistency, copy-rate (§6).

### 2.2 Harness
- `runner.py`: takes `(arm, knob_values, video)` → generates all shots → saves images under `runs/<stamp>/images/<arm>/<knob>/<shot_id>.png` → triggers the scoring pass at stage end (not per image) → appends JSONL records.
- `seeds.json`: one fixed seed per `shot_id`. Generate once, commit. Build `latents/<shot_id>.pt` from these seeds (`pipe.prepare_latents` shape) and pass via `latents=` in every run. Hash-check latents at load.
- Every generation logs `unet_calls` (count UNet forwards; with CFG, one denoising step = 1 batched call — count steps × micro-batches), `gen_time_s` (UNet loop only, model load excluded), `torch.cuda.max_memory_allocated()`.

### 2.3 Baseline reproduction — **gate G0**
Re-run the full DACA sweep on the **geology** video with the pinned seeds: every shot at every w in W (the w=0 render doubles as the cached content anchor for c_s, used by every later stage).

Pass criteria (in order of strictness, depending on whether the original thesis seeds are recoverable):
- If original seeds available: adaptive point within ±0.02 of the reported 0.339 / 0.790, fixed-scale table reproduced.
- If not (expected): qualitative reproduction — best fixed scale lands in the 0.3–0.5 region, DACA-adaptive c̄ exceeds best-fixed c̄ by ≥ +0.08 at ref-sim within ±0.02, same ordering pattern as the thesis. Also re-derive the τ-sweep **for free**: DACA selection is post-hoc on the logged c_s(w) curves, so compute the selected images and metrics for τ ∈ {0.5, 0.6, 0.7, 0.8, 0.9} from the same sweep without new generations. This is the baseline frontier.

Also at G0: print the **budget projection** (formulas in §7.3) using the measured s/generation and the actual shot counts from the storyboards. **Stop and report.**

---

## 3. Stage 1 — Alt-1: probe-then-commit (gate G1) — do this first, it is the low-risk path

**Semantics:** identical to DACA (anchoring active from step 0 at a fixed candidate scale; per-shot floor-argmax selection). Only the *cost* changes: candidates are run in parallel lanes for the first k steps only, ranked on cheap previews, and only the winner is completed.

### 3.1 Algorithm (per shot)
```
lanes = [0.0] + W                  # lane 0 anchors the preview-space content metric
lat0  = cached_latents[shot_id]    # identical tensor for every lane
k     = ceil(k_frac * num_steps)   # default k_frac = 0.30

for each micro-batch of lanes (chunk = 4, fallback 2 on OOM):
    state.mode = "per_sample"; state.w_per_sample = tensor(chunk_scales)
    state.s0 = 0.0                                  # DACA semantics: no timestep window here
    run denoising steps [0, k) from lat0 (replicated), stash latents at k

previews  = taesd_decode(latents_at_k)              # TAESD-SDXL, fp16
c_hat[w]  = DINO(preview_w, preview_0)              # reuse scoring_wrap's DINO with same preprocessing
w_star    = max{ w in W : c_hat[w] >= tau_prev }    # else min(W)
final     = continue denoising steps [k, N) from latents_at_k[w_star]; full VAE decode
```

### 3.2 Implementation notes
- **Pause/continue.** Two acceptable routes: (a) `StableDiffusionXLPipeline(..., denoising_end=k_frac, output_type="latent")` then `StableDiffusionXLImg2ImgPipeline.from_pipe(pipe)(..., denoising_start=k_frac, image=latents, ...)`; or (b) a minimal custom denoise loop (~40 lines) over `pipe.scheduler.timesteps`. Either way, the **mandatory equivalence check V-A1** below must pass.
- **Per-lane scales** require the custom processor from §5.2 in `per_sample` mode: the scale is a tensor of shape `[lanes]`, broadcast to `[2·lanes, 1, 1]` under CFG. Verify CFG batch ordering empirically (Appendix B, pitfall 1) — do not assume which half is unconditional.
- Prompt embeds and IP image embeds must be repeated to lane count; lanes are fully independent diffusions, so chunked micro-batches are exact, not an approximation.
- Lane 0 (w = 0) is needed for previews even though the full w=0 render is already cached — its *intermediate* latent at step k was never saved.

### 3.3 Verification & gate G1
- **V-A1 (pause/continue equivalence):** for one shot at one fixed w, a straight 0→N run and a pause-at-k-then-continue run must produce identical final latents (`torch.allclose`, fp16 tolerance). If this fails, the scheduler state handling is wrong — fix before anything else.
- **F1 (preview ranking validity)** — uses Stage-0 ground truth (full sweep on geology):
  - Per shot, Spearman rank correlation between ĉ(w) at step k and final c_s(w): **median over shots ≥ 0.8**.
  - Choose τ_prev on geology to maximize selection agreement with full-DACA selections; **agreement within ±1 grid index ≥ 80%** (also report exact-match rate).
  - End-to-end: Alt-1's adaptive frontier point within ±0.02 (both axes) of full DACA's on geology.
  - **Cost:** measured wall-clock ≤ 0.55 × full-DACA per shot at |W| = 6 (theoretical: (|W|+1)·k_frac + (1−k_frac) ≈ 2.8 generation-equivalents vs 6).
- **On fail:** retry k_frac ∈ {0.4, 0.5} (still cheaper than a full sweep). If still failing, **drop Alt-1**, record the rank-correlation curves in RESULTS.md, and fall back to full sweeps wherever a DACA-style selection is needed later.
- **G1: stop and report** F1 metrics + measured speedup. If F1 passes, Alt-1 becomes the default selection engine for any later arm that needs per-shot selection.

---

## 4. Stage 2 — Block-role re-probe under the cartoon LoRA (gate F2)

InstantStyle's published block roles (style ≈ `up_blocks.0.attentions.1`, layout ≈ `down_blocks.2.attentions.1`) were measured on base SDXL. The cartoon LoRA perturbs weights, so re-validate before trusting them. Budget: ~half a day, ~130 generations.

**Protocol**
- Probe set: 6 shot prompts (3 geology, 3 ecology) + their canonical references, fixed seeds from `seeds.json`.
- Injection sites (single-site = layer gate 1.0 there, 0.0 everywhere else): `global` (all attn2 — control), `down2.att0`, `down2.att1`, `mid.att0`, `up0.att0`, `up0.att1`, `up0.att2`.
- Scales w ∈ {0.3, 0.5, 0.8} per site, plus the shared w=0 renders. Single canonical reference (no centroid yet), no timestep window (s0 = 0).
- For every render compute c_s (vs w=0) and ref_sim. Save a contact-sheet grid per site (`blockprobe/site_<name>.png`).

**Analysis & gate F2**
- Per site, plot the (c_s, ref_sim) curve over w, averaged over the 6 prompts. Select `STYLE_LAYERS` = the site(s) whose curve **Pareto-dominates the `global` curve** (higher ref_sim at matched c, or higher c at matched ref_sim). Expected winner: `up0.att1`; trust the data over the expectation.
- **If no single site dominates global:** block targeting is dropped (layer gates revert to global). Then run the **window mini-gate**: with global gates, s0 = 0.3 must improve c̄ by ≥ +0.05 at matched ref_sim (re-tune w to match) vs s0 = 0 on the probe set. If that also fails, component 2 is dropped entirely — record both negative results honestly; FACET reduces to centroid (+ controller).
- Deliverable: `blockprobe/REPORT.md` with grids, curves, chosen `STYLE_LAYERS`, and the F2 verdict. (Bundle into the G1 or G3 report — no separate human stop needed unless F2 fails.)

---

## 5. FACET components (Phase-4 hooks only)

All three components live in `pipeline/facet/` and attach to the *existing, unmodified* pipeline object via attention-processor replacement and a step callback. No weight changes, no training.

### 5.1 Centroid concept embedding (`centroid.py`)

**Build (one-time per concept, cached):**
1. Generate the 5 variants (Stage 3) with the concept prompt + LoRA, seeds 1001–1005. Eyeball the contact sheet — discard and reseed any degenerate variant.
2. **Version-robust embedding recipe** — do not assume tensor layouts:
   - Call the pipeline's own `prepare_ip_adapter_image_embeds([img], ...)` once per variant and **print the shapes** (list length per adapter, neg/pos stacking convention, token dims). Mirror exactly that layout when reassembling.
   - **Vanilla IP-Adapter** (global embed): `centroid_pos = Σ a_k · e_k / Σ a_k` with weights a = 2 for the canonical, 1 for each variant. Keep the negative half (zeros) unchanged. This is the clean, preferred case.
   - **IP-Adapter Plus** (patch tokens, e.g. [*, 257, dim]): default **Option A** = the same weighted mean per token position across variants (patch positions are only roughly comparable across images — that is exactly what V-C2 checks). **Option B (fallback)** = wrap `unet.encoder_hid_proj` to project each variant through the Resampler separately and average the 16 output tokens per slot (slots come from fixed learned queries, so they are in better correspondence). Use B only if A fails V-C2.
   - Optional refinement (Plus only, only if A passes weakly): token-stability weighting — per-token weights ∝ 1/(1 + Var across variants), renormalized.
3. Cache as `refs/variants/<concept_id>/centroid_embeds.pt`. At generation time pass via `ip_adapter_image_embeds=[...]` instead of `ip_adapter_image=`.

**Verification V-C2 / gate F3:**
- **Plumbing check:** feeding the *single canonical* embedding through the new `ip_adapter_image_embeds` path must reproduce the baseline render of a test shot (DINO cosine ≥ 0.999 vs the `ip_adapter_image=` path; ideally allclose latents).
- **Effect check** (6 probe shots, fixed mid-w, global gates, s0 = 0 — isolate the centroid): vs single-reference, require **c̄ +0.03 or pairwise-consistency +0.02**, AND pairwise drop ≤ 0.02, AND no visual mush on the contact sheet.
- **On fail:** retry with medoid + shrinkage (embed = 0.5·medoid + 0.5·centroid, medoid = variant with max mean cosine to the others). If still failing, **drop the centroid** (F3) and keep the single canonical embedding everywhere.

### 5.2 Targeted injection — `FacetState` + `FacetIPAttnProcessor` (`processors.py`)

Fork the installed diffusers version's `IPAdapterAttnProcessor2_0` source (copy it into the repo so the diff is explicit) and apply four edits. Reuse the already-loaded `to_k_ip` / `to_v_ip` modules from the processor instance being replaced — never re-initialize them.

```python
class FacetState:
    """Single shared mutable object; read by every FacetIPAttnProcessor, updated by the step callback."""
    mode: str                  # "fixed" | "controller" | "per_sample"(Alt-1)
    w_fixed: float             # fixed-w arms
    w_per_sample: Tensor|None  # [num_lanes], Alt-1
    rho_star: float            # controller target
    w_current: float           # controller output (also logged)
    r_ema: float|None          # EMA of median(o_txt / o_img), alpha = 0.3
    frac: float                # progress in [0,1); step i uses frac = i / N
    s0: float = 0.30           # timestep window start (fraction of steps)
    ramp: float = 0.10         # cosine ramp width
    layer_gate: dict[str,float]# processor_name -> gate (1.0 for STYLE_LAYERS, else 0.0;
                               #   optional comp-leak: gate = 0.2 on down2.att1, config flag, default OFF)
    step_norms: dict[str,tuple]# processor_name -> (||O_txt||, ||O_img||) for the current step
    trace: list                # [(step, w_eff, rho_realized_median)]

    def time_gate(self) -> float:
        if self.frac < self.s0: return 0.0
        x = min(1.0, (self.frac - self.s0) / max(self.ramp, 1e-6))
        return 0.5 - 0.5 * math.cos(math.pi * x)
```

The four edits to the forked processor:
1. `__init__(self, name, state, to_k_ip, to_v_ip, ...)` — store the processor's own `name` and the shared `state`; register the existing IP projection modules (same dtype/device).
2. **Always compute the IP branch**, even when the effective scale is 0. Needed so norms exist during the text-only window (controller warm start). The extra cost is small and uniform across arms.
3. After the text branch `hidden_states` and the image branch `ip_hidden_states` are computed (both **pre-`to_out`**):
   ```python
   B = batch // 2 if doing_cfg else 0          # conditional half = [B:]; verify ordering (Appendix B.1)
   state.step_norms[self.name] = (hidden_states[B:].float().norm().item(),
                                  ip_hidden_states[B:].float().norm().item())
   ```
4. Effective scale and merge:
   ```python
   g = state.layer_gate.get(self.name, 0.0) * state.time_gate()
   w = {"fixed": state.w_fixed,
        "controller": state.w_current,
        "per_sample": state.w_per_sample.repeat(2)[:, None, None]  # match CFG batch layout
       }[state.mode]
   hidden_states = hidden_states + (g * w) * ip_hidden_states
   ```

Installer:
```python
def install_facet(pipe, state, style_layer_substrings):
    procs, new = pipe.unet.attn_processors, {}
    for name, p in procs.items():
        if hasattr(p, "to_k_ip"):                              # IP-Adapter cross-attn processor
            new[name] = FacetIPAttnProcessor(name, state, p.to_k_ip, p.to_v_ip, ...)
            state.layer_gate[name] = 1.0 if any(s in name for s in style_layer_substrings) else 0.0
        else:
            new[name] = p
    pipe.unet.set_attn_processor(new)
    state.targeted = [n for n, g in state.layer_gate.items() if g > 0]
```
After installation, the runner must **never call `pipe.set_ip_adapter_scale`** (it would mutate or replace processors — pitfall B.3). For Alt-1, install with `style_layer_substrings = ["attn2"]`-equivalent global gating and `s0 = 0`.

**Step callback** (registered via `callback_on_step_end`; runs *after* step i):
```python
def facet_callback(pipe, i, t, kw, *, state, num_steps):
    if state.mode == "controller":
        ratios = [ot / max(oi, 1e-8) for ot, oi in (state.step_norms[n] for n in state.targeted)]
        r = float(np.median(ratios))
        state.r_ema = r if state.r_ema is None else 0.7 * state.r_ema + 0.3 * r
        w_hat = state.rho_star / (1.0 - state.rho_star) * state.r_ema
        state.w_current = float(np.clip(w_hat, 0.05, 0.90))
        rho_real = float(np.median([state.w_current*oi/(state.w_current*oi+ot)
                                    for ot, oi in (state.step_norms[n] for n in state.targeted)]))
        state.trace.append((i, state.w_current, rho_real))
    state.step_norms.clear()
    state.frac = (i + 1) / num_steps
    return kw
```
(The callback fires after step i, so the w it sets is consumed at step i+1 — a one-step lag the EMA absorbs. During the text-only window the IP branch is computed but unapplied, so `r_ema` warm-starts and `w_current` is already sensible when the ramp opens.)

### 5.3 Norm-share controller (`controller.py` — thin config layer over 5.2)

Definitions, per targeted layer ℓ at step t (cond half, pre-`to_out`, Frobenius norms):
- realized influence share ρ_ℓ = w·‖O_img,ℓ‖ / (w·‖O_img,ℓ‖ + ‖O_txt,ℓ‖)
- closed-form setpoint solve: ŵ_t = ρ* / (1 − ρ*) · median_ℓ( ‖O_txt,ℓ‖ / ‖O_img,ℓ‖ ), smoothed via the single EMA on the ratio (α = 0.3), clamped to [0.05, 0.90].

Defaults: ρ* grid {0.10, 0.15, 0.20, 0.30, 0.40}. Log the full per-step trace (step, w, realized ρ) for every controller shot — P3/F4 and the stability gates need it.

**Stability gates (per shot, evaluated after the ramp opens):**
- std of step-to-step Δw ≤ 0.05;
- clamp railing (w pinned at 0.05 or 0.90) on ≤ 30% of anchored steps, for ≤ 30% of shots.
On violation: halve α to 0.15 and rerun the offending shots once. If violations persist, mark the controller **unstable** — this feeds F4 (§8) directly.

---

## 6. Metrics & logging

### 6.1 What is measured (all DINO/CLIP calls go through `scoring_wrap.py` — the repo's own preprocessing)

| Metric | Definition | Notes |
|---|---|---|
| `c_s` (content preservation) | DINOv2 cosine(shot @ config, same shot @ w=0) | same shot_id, same cached latents; the w=0 renders come from Stage 0 and are reused everywhere |
| `ref_sim` | DINOv2 cosine(shot, **canonical** reference image) | always vs the canonical, in every arm — even when the *injected* embedding is a centroid. The canonical is the fixed yardstick for comparability with thesis numbers |
| `pairwise` (cross-shot consistency) | mean DINOv2 cosine over all unordered same-concept shot pairs | **exclude flagged near-reference control shots** (config list; includes the known ecology control). Skip concepts with < 2 eligible shots |
| `clip_t` | CLIP similarity(shot image, the shot's narration/visual-prompt text) | exactly the text field the thesis used |
| `copy_rate` | fraction of shots with ref_sim ≥ θ_copy | report at θ ∈ {0.80, 0.85, 0.90}; headline θ = 0.85, recalibrated at G0 to sit just below the ecology control shot's measured ref_sim |
| cost | `gen_time_s` (UNet loop wall-clock), `unet_calls`, `vram_peak_gb` | per shot and per arm totals; `unet_calls` is the hardware-independent number |

Aggregates (`c̄`, mean ref_sim, pairwise, CLIP-T) are reported per concept and per video, **both including and excluding** the flagged control shots.

### 6.2 JSONL record (one line per generated image)
```
run_id, stage, arm, video, concept_id, shot_id, seed,
knobs: { w | rho_star, s0, ramp, layer_gates_id, K, centroid_mode, tau | tau_prev, k_frac },
unet_calls, gen_time_s, vram_peak_gb,
w_trace: [[step, w, rho_realized], ...]        # controller arms only
metrics: { c_s, ref_sim, clip_t },             # filled by the scoring pass
paths: { image, latents }
```
Scoring is a **separate pass at stage end**: unload SDXL → load DINOv2 → embed every new image once → cache embeddings (`embeds/<shot>.pt`) → CLIP pass → fill metrics. Pairwise, copy-rate and aggregates are computed by `aggregate.py` from cached embeddings.

### 6.3 Frontier plots (`aggregate.py`)
Per video, two plots, one curve per arm over its knob grid:
- x = mean ref_sim, y = c̄;
- x = pairwise, y = c̄.
DACA contributes two baseline curves from the Stage-0 sweep: the fixed-w curve and the τ-sweep curve (τ ∈ {0.5…0.9}, recomputed post-hoc, zero extra generations). Output a dominance table: for each non-baseline arm, does any knob setting Pareto-dominate any baseline point / is the arm dominated anywhere.

---

## 7. Ablation arms, work order, budget

### 7.1 Arms
| id | injected ref | layer gates | window s0 | scale source | knob grid | videos |
|---|---|---|---|---|---|---|
| A0 | canonical | global | 0 | fixed-w sweep + DACA/τ post-hoc | W; τ ∈ {0.5…0.9} | geology (G0), ecology (Stage 4) |
| A1 | canonical | global | 0 | Alt-1 probe-then-commit | τ_prev, k_frac | geology (validation), then both |
| B1 | **centroid** | global | 0 | fixed w | {0.2, 0.3, 0.4, 0.5, 0.6} | both |
| B2 | canonical | **STYLE_LAYERS** | **0.30** | fixed w | {0.2, 0.3, 0.4, 0.6, 0.8}* | both |
| B3 | centroid | STYLE_LAYERS | 0.30 | fixed w | as B2 | both |
| C2 | canonical | global | 0 | **controller** | ρ* ∈ {0.10…0.40} | both |
| C1 = FACET | centroid | STYLE_LAYERS | 0.30 | controller | ρ* ∈ {0.10…0.40} | both |
| D (optional) | centroid | STYLE_LAYERS | 0.30 | DACA selection (via Alt-1 if F1 passed) | τ = 0.7 | geology only |

*B2/B3 grids extend to 0.8 because gating fewer layers lowers effective strength at the same nominal w.

C2 exists purely to give P3 a clean test on the vanilla pipeline (no confound from centroid/targeting). D holds the *selection rule* fixed while changing the pipeline — the cleanest single demonstration of frontier bending; run it only if B3 looks good and budget allows.

### 7.2 Work order (cheapest/safest first) and stage gates
1. **Stage 0** — recon + harness + A0 on geology. Gate **G0** (reproduction + budget projection). *Stop, report.*
2. **Stage 1** — Alt-1 (A1) on geology vs Stage-0 ground truth. Gates V-A1, **F1**. *Stop, report (G1).*
3. **Stage 2** — block probe. Gate **F2** → `STYLE_LAYERS` (or drop decision). Cheap; bundle report with G1 or G3.
4. **Stage 3** — variant generation + centroid build. Gates plumbing check + V-C2/**F3** (probe shots only).
5. **Stage 4** — A0 on ecology; B1/B2/B3 frontiers on both videos. **Gate G3:** the frontier-bend verdict — does B3 Pareto-improve on the baseline frontier on ≥ 1 video without being dominated anywhere? This is the primary scientific result; *stop, report,* with plots.
6. **Stage 5** — C2, then C1. Gates: stability, **F4 = P3**. Frontier comparison vs B3 and baselines.
7. **Stage 6** — `aggregate.py` over everything; P1–P5 table; choose ship config by §8; Phase-5 I2V for ship config + DACA on one video. **Gate G4:** final report. *Stop.*

Rationale: A1 is a near-guaranteed engineering win and de-risks every later selection step; the probe and centroid checks are < 1 day each and gate the expensive frontier sweeps; the controller goes last because it depends on B3's pipeline and is the component most likely to be dropped.

### 7.3 Budget formulas (fill in at G0 with measured s/gen and shot counts)
Let N_g, N_e = shots per video, N = N_g + N_e, |W| = 6, |grid_B| = 5, |grid_ρ| = 5.
- A0: N · |W| generations (geology half already done at G0; w=0 renders included in W's sweep or +N if w=0 ∉ W).
- A1 validation: N_g · ((|W|+1)·k_frac + (1−k_frac)) ≈ 2.8 · N_g gen-equivalents.
- Block probe: 6 prompts · 7 sites · 3 scales + 6 ≈ 132. Variants: 5 · n_concepts.
- B1+B2+B3: 3 · N · |grid_B|. C2+C1: 2 · N · |grid_ρ|. D: N_g · (Alt-1 cost).
Everything fits a 3090 in low single-digit GPU-days at ~10 s/gen; print the actual table at G0.

---

## 8. Falsifiable checks & drop rules

| ID | Claim under test | Test | Pass threshold | On fail |
|---|---|---|---|---|
| **F1** | Early previews rank scales like final images | §3.3 vs Stage-0 ground truth | median Spearman ≥ 0.8; ±1-index selection agreement ≥ 80%; frontier point within ±0.02 | k_frac ∈ {0.4, 0.5}; else **drop Alt-1**, use full sweeps |
| **F2** | Style/content blocks are separable under the cartoon LoRA | §4 probe | ≥ 1 site Pareto-dominates global on the probe set | targeting → global; window mini-gate (c̄ +0.05 at matched ref_sim); else **drop component 2** |
| **F3** | Centroid anchors concept without blurring it | V-C2 (§5.1) | c̄ +0.03 or pairwise +0.02, pairwise drop ≤ 0.02, plumbing check passes | medoid + shrinkage; else **drop centroid** |
| **F4 = P3** | Equalizing *realized* influence reduces per-shot content variance vs equal *nominal* w | see below | Var ratio ≤ 0.5 on the majority of matched pairs, **required in C1-vs-B3** (C2-vs-A0 is informative) | **drop controller**; FACET ships as B3 + Alt-1 selection (if F1 passed) else fixed-w chosen per video on dev |

**P3 procedure (the user's headline check):** for each ρ* in the grid, find the fixed-w setting (same pipeline: C2↔A0-fixed, C1↔B3) whose mean c̄ over the same shot set is within ±0.02 — these are matched operating points; require ≥ 3 matched pairs across the grid. For each pair compute Var across shots of c_s. Pass iff Var(c_s | ρ*) ≤ 0.5 · Var(c_s | matched w) for the majority of pairs in the C1-vs-B3 comparison, **and** the §5.3 stability gates hold. Report the full table either way; a clean negative here is a publishable finding about the norm-share proxy.

**Always-on guard:** any config with copy_rate(θ = 0.85) > baseline DACA's + 5 points is excluded from all frontier claims, regardless of how good its curves look.

**Final decision rules (pre-registered; directional only at n = 2 videos):**
- **P1** (frontier bend): at matched ref_sim (±0.01, interpolate along curves), best FACET-family arm achieves c̄ ≥ DACA + 0.05 on ≥ 1 of 2 videos and ≥ DACA − 0.02 on the other. (Criterion graduates to "≥ 3 of 4–5 videos, never dominated" when the dataset expands.)
- **P2**: pairwise and CLIP-T no worse than DACA at the chosen operating point (tolerance −0.01).
- **P4** (cost): per-shot *selection* cost ≥ 3× cheaper than the full DACA sweep, via Alt-1 and/or the controller path; report measured wall-clock and unet_calls.
- **P5**: copy-rate guard holds.
- **Expected and acceptable side effect:** centroid arms may show slightly *lower* ref_sim-to-canonical while pairwise holds or improves — judge those arms on the pairwise plot too; this is the thesis's own metric-trap argument applied consistently.

---

## 9. Deliverables

```
pipeline/facet/{processors.py, controller.py, centroid.py, alt1.py, runner.py,
                scoring_wrap.py, aggregate.py}
configs/facet.yaml
runs/<stamp>/{records.jsonl, images/, embeds/, latents/, blockprobe/, plots/, RESULTS.md}
```
`RESULTS.md` template: environment + hashes; G-gate table; F-gate table with verdicts and the evidence plot for each; frontier plots per video with dominance tables; P1–P5 table; chosen ship config; anomalies and any deviation from this brief (every deviation must be listed).

---

## Appendix A — defaults (single source of truth: `configs/facet.yaml`)

| Knob | Default | Notes |
|---|---|---|
| K (refs incl. canonical) | 6 | canonical weight 2, variants 1 |
| s0 / ramp | 0.30 / 0.10 | cosine ramp; no end taper in v1 |
| controller clamp / α | [0.05, 0.90] / 0.3 | α → 0.15 on instability retry |
| ρ* grid | {0.10, 0.15, 0.20, 0.30, 0.40} | |
| fixed-w grids | B1: {0.2…0.6}; B2/B3: {0.2…0.8} | §7.1 |
| k_frac / probe chunk | 0.30 / 4 (fallback 2) | Alt-1 |
| τ (DACA) / τ_prev | 0.7 / calibrated in F1 | |
| θ_copy headline | 0.85 | recalibrated at G0 vs ecology control shot |
| comp-leak gate | OFF | enable per-concept (0.2) only if Stage 4 shows a diagram-like concept failing consistency; log when used |
| steps / CFG / resolution / scheduler / W | **read from existing DACA config — never change** | |

## Appendix B — known pitfalls (check each one explicitly)

1. **CFG batch ordering.** Diffusers conventionally concatenates [uncond, cond], but verify empirically: change the prompt, observe which batch half's activations move, and slice norms/per-sample scales accordingly.
2. **`prepare_ip_adapter_image_embeds` layout varies across diffusers versions.** Mirror the observed layout (the plumbing check in §5.1 exists to catch this); never hard-code shapes.
3. **`set_ip_adapter_scale` clobbers custom processors** (it mutates/installs processor state). After `install_facet`, all scaling goes through `FacetState` only.
4. **Ancestral schedulers** inject per-step noise → pause/continue and latent caching break. Deterministic scheduler everywhere (§0.5); V-A1 catches violations.
5. **LoRA fuse state** must be identical across arms (fused vs unfused changes outputs slightly). Record it at G0 and assert it per run.
6. **Always compute the IP branch even at gate 0** (edit 2 in §5.2) — otherwise the controller has no warm start and the text-only window logs nothing.
7. **One-model residency:** explicit unload + `empty_cache()` between generation and scoring passes; assert free VRAM before loading the next model. TAESD is the only sanctioned co-resident.
8. **Latents cache:** fp16, CUDA, identical tensor per shot across all arms — hash-check at load. If a latents file is missing, *stop*; do not silently regenerate with a new seed.
9. **Control shots:** keep them in generation, exclude them from `pairwise`, and report aggregates both including and excluding them (§6.1).
10. **Norm logging overhead:** `.item()` per layer per step forces syncs; if profiling shows > 3% overhead, accumulate norms on-GPU and sync once per step in the callback.
