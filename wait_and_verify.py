import time
import json
import numpy as np
from pathlib import Path
from PIL import Image

def wait_for_files():
    base = Path("data/intermediate/lT_QAkL6lj0_where-do-rocks-come-from-crash-course-ge/phase4")
    sample_dir = base / "_sample_video"
    
    files = [
        "semantic_triggered_shot_001_I2V.webp",
        "semantic_triggered_shot_004_FLF2V.webp",
        "semantic_triggered_shot_010_I2V.webp"
    ]
    
    print("Waiting for generation to finish...")
    for f in files:
        p = sample_dir / f
        while not p.exists() or p.stat().st_size < 1024:
            time.sleep(30)
            
    print("All files generated! Running verification...")
    
def calculate_mse(imageA, imageB):
    err = np.sum((imageA.astype("float") - imageB.astype("float")) ** 2)
    err /= float(imageA.shape[0] * imageA.shape[1])
    return err

def verify_mode(clip_path: Path, expected_first_frame: Path, expected_last_frame: Path = None):
    img = Image.open(clip_path)
    frames = []
    
    for i in range(img.n_frames):
        img.seek(i)
        frames.append(img.convert("RGB"))
        
    actual_first = frames[0]
    actual_last = frames[-1]
    
    expected_first = Image.open(expected_first_frame).convert("RGB")
    expected_first = expected_first.resize(actual_first.size)
    mse_first = calculate_mse(np.array(actual_first), np.array(expected_first))
    
    res = {"frames": img.n_frames, "size": actual_first.size, "mse_first": float(mse_first)}
    
    if expected_last_frame:
        expected_last = Image.open(expected_last_frame).convert("RGB")
        expected_last = expected_last.resize(actual_last.size)
        mse_last = calculate_mse(np.array(actual_last), np.array(expected_last))
        res["mse_last"] = float(mse_last)
        
    return res

if __name__ == "__main__":
    wait_for_files()
    
    base = Path("data/intermediate/lT_QAkL6lj0_where-do-rocks-come-from-crash-course-ge/phase4")
    sample_dir = base / "_sample_video"
    results = {}
    
    clip1 = sample_dir / "semantic_triggered_shot_001_I2V.webp"
    ref_start1 = base / "semantic_triggered/images/shot_001.png"
    results["shot_001"] = verify_mode(clip1, ref_start1)
    
    clip2 = sample_dir / "semantic_triggered_shot_004_FLF2V.webp"
    ref_start2 = Path("/tmp/video_summarizer_frames")
    import glob
    ff = glob.glob("/tmp/video_summarizer_frames/*shot_003*last_frame.png")
    if len(ff) > 0:
        ref_start2 = Path(ff[0])
    else:
        ref_start2 = base / "semantic_triggered/images/shot_003.png" # fallback mock
    ref_end2 = base / "semantic_triggered/images/shot_004.png"
    results["shot_004"] = verify_mode(clip2, ref_start2, ref_end2)
    
    clip3 = sample_dir / "semantic_triggered_shot_010_I2V.webp"
    ref_start3 = base / "semantic_triggered/images/shot_010.png"
    results["shot_010"] = verify_mode(clip3, ref_start3)
    
    with open("/tmp/verification_results.json", "w") as f:
        json.dump(results, f, indent=2)
    print("Verification complete.")
