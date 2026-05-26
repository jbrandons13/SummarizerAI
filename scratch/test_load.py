import os
from diffusers import LTXImageToVideoPipeline
import torch

try:
    print("Checking loading LTXImageToVideoPipeline from HF hub id...")
    pipe = LTXImageToVideoPipeline.from_pretrained(
        "Lightricks/LTX-Video-0.9.7-distilled",
        torch_dtype=torch.bfloat16
    )
    print("LTX Pipeline loaded successfully from Hugging Face hub ID.")
    print("Pipeline class:", pipe.__class__)
except Exception as e:
    print("Failed loading from HF hub ID:", e)
    # Check local paths
    local_paths = [
        "/home/wins053/models/ltx_video_distilled",
        "/home/wins053/models/ltx-video-0.9.7-distilled",
        "./models/ltx_video_distilled"
    ]
    for lp in local_paths:
        if os.path.exists(lp):
            print(f"Found local path: {lp}. Attempting to load...")
            try:
                pipe = LTXImageToVideoPipeline.from_pretrained(
                    lp,
                    torch_dtype=torch.bfloat16
                )
                print("LTX Pipeline loaded successfully from local path:", lp)
                break
            except Exception as ex:
                print(f"Failed loading from {lp}: {ex}")
