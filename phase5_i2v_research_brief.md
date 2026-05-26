# I2V Model Research Brief: SOTA Candidates for Subject-Preserving I2V on Consumer GPU

## Context

Previous Phase 5 smoke tests showed:
- **Wan 2.2 TI2V-5B**: catastrophic drift (fade to brown across all 3 inputs).
- **CogVideoX-5B I2V**: near-static output (slow zoom/pan only, no meaningful motion).
- **FLUX.1-schnell + XLabs IP-Adapter**: IP-Adapter had no observable effect; output followed prompt only, no anchor to conditioning frame.

We need an I2V or image-conditioned video generation model that:
1. **Preserves subject identity** strongly from a conditioning frame (the gadget being reviewed must remain visually consistent — same color, shape, brand markings, etc.).
2. **Fits on RTX 3090 (22 GB usable VRAM)** with reasonable inference time, ideally under 5 minutes per ~3-6 second clip.
3. **Has mature HuggingFace Diffusers integration** (or trivial-to-install custom code). Time is constrained — we cannot afford 3-5 days debugging custom training/inference code.
4. **Is open source** with non-restrictive license (Apache 2.0, FLUX-dev non-commercial OK for academic thesis, etc.).

This brief asks for a research-and-recommendation pass. Do NOT run any smoke tests in this task. Output is a written recommendation we can act on next.

## Candidates to investigate

Investigate at least these, but feel free to add any newer model you find that fits the criteria:

### Priority candidates (recent, claimed consumer-GPU friendly)

1. **HunyuanVideo 1.5** (Tencent, released Nov 2025). 8.3B parameters. Claimed to enable practical inference on consumer GPUs at 480p I2V. Check:
   - Is the I2V variant available on HuggingFace?
   - Diffusers support status (native pipeline or community)?
   - Actual VRAM at 480p I2V on RTX 3090 (24 GB)?
   - Reported subject identity preservation quality vs competitors?
   - Inference time per second of output?

2. **LTX-2 / LTX-Video 2** (Lightricks, released Jan 2026). 19B (14B video + 5B audio). 4K native, NVFP8 quantization. Check:
   - VRAM footprint with quantization at 480p/720p output?
   - Is there a smaller I2V variant suitable for 22 GB?
   - Subject preservation rating (community testing notes suggest LTX-Video v1 struggled with detailed identity over multiple seconds — has v2 improved this?)
   - Apache 2.0 license confirmed?

3. **Wan 2.2 I2V-A14B** (Mixture-of-Experts 14B variant of what we already tested). Different from the 5B we tried. Check:
   - VRAM with `enable_model_cpu_offload` and `enable_sequential_cpu_offload` on RTX 3090?
   - Inference time per 3-second clip with offloading?
   - Diffusers support status?
   - Is identity preservation actually better than 5B at our hardware budget, or do we hit OOM / unacceptable latency?

### Lower-priority but worth a paragraph each

4. **Mochi 1** (Genmo, 10B, AsymmDiT, Apache 2.0). I2V mode supported? Quality vs Wan 2.2?
5. **HunyuanVideo Avatar** (mentioned in research as having superior identity preservation vs base HunyuanVideo I2V). Is this only for talking-head / face use cases, or general subjects? Probably unsuitable for gadget review domain — confirm and move on if so.
6. **Waver 1.0** (mentioned as unified DiT supporting T2V, I2V, T2I up to 1080p). Open source? Hardware reqs?

### Fallback candidates (already known, list briefly for comparison)

7. **FLUX.1-dev img2img** (no IP-Adapter, just diffusion img2img with the retrieved frame as init latent). Not state-of-the-art video, but extremely reliable for identity preservation. Include as baseline option.
8. **Stable Video Diffusion 1.1** (older, 8-16 GB VRAM, lower quality ceiling). Include as ultra-safe fallback only.

## Required output

A markdown document, 1-2 pages, with these sections:

### Section A: Per-candidate factsheet

For each priority candidate (1-3) and a brief paragraph for each lower-priority (4-6), report:
- Exact HuggingFace repo path (e.g., `tencent/HunyuanVideo-1.5-I2V` — verify this exists).
- Parameter count and architecture summary.
- Required VRAM at lowest viable resolution, with and without offloading.
- Diffusers pipeline class name if native support exists, or note the integration path otherwise.
- License.
- Release date.
- Community-reported strengths and weaknesses, especially around subject identity preservation.
- Any reported quirks (e.g., needs specific scheduler, prompt template, frame preprocessing).

### Section B: Recommendation

A single recommendation: which model should we smoke test first, and why. Rank top 3.

The ranking criterion is **time to confident go/no-go decision**, not just raw quality. A slightly-lower-quality model with mature Diffusers integration beats a higher-quality model that needs custom inference code.

### Section C: Risks per candidate

For the top recommendation, list:
- Likely failure modes (OOM, slow inference, poor identity preservation, prompt sensitivity, etc.).
- Download size and bandwidth cost (we have ~20 GB/session quota, somewhat flexible).
- Estimated smoke test execution time (model download + 3-input inference run).

## Constraints

- This is a research task. Do NOT run any inference, do NOT download any models, do NOT modify any code.
- Verify HuggingFace repo paths exist before listing them. If you cite a model, confirm it is actually accessible on HuggingFace as of today.
- If a model card or community report contradicts marketing claims (e.g., a paper claims X but issues on the repo report Y), flag the discrepancy.
- If for any candidate the information is sparse or unreliable, mark as "INSUFFICIENT DATA — needs hands-on smoke test to verify."
- Output the markdown file to: `phase5_i2v_model_research.md` in the repo root.

=== END OF BRIEF ===
