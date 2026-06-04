import json
from pathlib import Path
from PIL import Image, ImageChops, ImageStat
import numpy as np

def calculate_mse(imageA, imageB):
    err = np.sum((imageA.astype("float") - imageB.astype("float")) ** 2)
    err /= float(imageA.shape[0] * imageA.shape[1])
    return err

def verify_mode(clip_path: Path, expected_first_frame: Path, expected_last_frame: Path = None):
    img = Image.open(clip_path)
    frames = []
    
    # Extract frames
    for i in range(img.n_frames):
        img.seek(i)
        frames.append(img.convert("RGB"))
        
    actual_first = frames[0]
    actual_last = frames[-1]
    
    expected_first = Image.open(expected_first_frame).convert("RGB")
    expected_first = expected_first.resize(actual_first.size)
    
    mse_first = calculate_mse(np.array(actual_first), np.array(expected_first))
    print(f"First frame MSE: {mse_first:.2f}")
    
    if expected_last_frame:
        expected_last = Image.open(expected_last_frame).convert("RGB")
        expected_last = expected_last.resize(actual_last.size)
        mse_last = calculate_mse(np.array(actual_last), np.array(expected_last))
        print(f"Last frame MSE: {mse_last:.2f}")

if __name__ == "__main__":
    base = Path("data/intermediate/lT_QAkL6lj0_where-do-rocks-come-from-crash-course-ge/phase4")
    sample_dir = base / "_sample_video"
    
    # Verify shot_001 (I2V)
    print("\n--- Verifying shot_001 (I2V) ---")
    clip = sample_dir / "semantic_triggered_shot_001_I2V.webp"
    ref_start = base / "semantic_triggered/images/shot_001.png"
    if clip.exists():
        verify_mode(clip, ref_start)
    else:
        print("Clip not found.")
        
    # Verify shot_004 (FLF2V)
    print("\n--- Verifying shot_004 (FLF2V) ---")
    clip = sample_dir / "semantic_triggered_shot_004_FLF2V.webp"
    ref_start = Path("/tmp/video_summarizer_frames") # Will be dynamically populated
    # wait, we need the exact frame path. For now, just a placeholder.
    # The actual Sub-B verification is meant to be visual by the user or via PIL.
