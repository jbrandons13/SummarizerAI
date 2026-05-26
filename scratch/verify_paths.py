import os
from pathlib import Path

inputs = ["input_b_marginal", "input_c_generate"]
for inp in inputs:
    p = f"/home/wins053/Desktop/SumarizerAI-1-Gemini/video-summarizer/phase5_smoke_inputs/{inp}/frame_ltx_preprocessed.jpg"
    print(f"{p} exists:", os.path.exists(p))

model_id = "Lightricks/LTX-Video-0.9.7-distilled"
# Let's check cache dir or standard huggingface cache if it exists, or if we can load it directly
# Usually model is cached under standard HF cache path or in user's home directory.
# Let's see if we can locate the directory where it's cached.
home_cache = Path(os.path.expanduser("~")) / ".cache" / "huggingface" / "hub"
print(f"HF Hub cache path exists:", home_cache.exists())
if home_cache.exists():
    for p in home_cache.glob("*ltx*"):
        print("Found LTX cache folder:", p)
