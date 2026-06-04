from PIL import Image
import numpy as np
from pathlib import Path
import json

def get_frames(path):
    img = Image.open(path)
    frames = []
    for i in range(img.n_frames):
        img.seek(i)
        frames.append(np.array(img.convert("RGB")))
    return np.array(frames)

base = Path("data/intermediate/lT_QAkL6lj0_where-do-rocks-come-from-crash-course-ge/phase4")
smoke = base / "_smoke_test/shot_010_FLF2V_test.webp"
regression = base / "_regression_test/shot_010_FLF2V_regression.webp"

if smoke.exists() and regression.exists():
    f1 = get_frames(smoke)
    f2 = get_frames(regression)
    
    err = np.sum((f1.astype("float") - f2.astype("float")) ** 2)
    err /= float(f1.shape[0] * f1.shape[1] * f1.shape[2])
    print(f"MSE: {err:.2f}")
    
    with open("/tmp/regression_mse.txt", "w") as f:
        f.write(f"MSE: {err:.2f}\n")
else:
    print("Files missing")
