import os
from PIL import Image

base_dir = "data/intermediate/lT_QAkL6lj0_where-do-rocks-come-from-crash-course-ge/phase4"
never_chain_dir = os.path.join(base_dir, "never_chain", "images")
w01_dir = os.path.join(base_dir, "concept_anchor_canonical_w01", "images")
w02_dir = os.path.join(base_dir, "concept_anchor_canonical_w02", "images")
w03_dir = os.path.join(base_dir, "concept_anchor_canonical_w03", "images")
w04_dir = os.path.join(base_dir, "concept_anchor_canonical_w04", "images")
out_dir = os.path.join(base_dir, "_eval", "montages_cross_weight_low")
os.makedirs(out_dir, exist_ok=True)

def make_cross_weight_low_montage(shot_id, anchor_id):
    img_baseline = Image.open(os.path.join(never_chain_dir, f"{shot_id}.png"))
    img_w01 = Image.open(os.path.join(w01_dir, f"{shot_id}.png"))
    img_w02 = Image.open(os.path.join(w02_dir, f"{shot_id}.png"))
    img_w03 = Image.open(os.path.join(w03_dir, f"{shot_id}.png"))
    img_w04 = Image.open(os.path.join(w04_dir, f"{shot_id}.png"))
    img_anchor = Image.open(os.path.join(never_chain_dir, f"{anchor_id}.png"))
    
    w, h = img_baseline.size
    # 6 columns
    montage = Image.new("RGB", (w * 6, h))
    montage.paste(img_baseline, (0, 0))
    montage.paste(img_w01, (w, 0))
    montage.paste(img_w02, (w * 2, 0))
    montage.paste(img_w03, (w * 3, 0))
    montage.paste(img_w04, (w * 4, 0))
    montage.paste(img_anchor, (w * 5, 0))
    
    out_path = os.path.join(out_dir, f"cross_weight_low_{shot_id}_to_{anchor_id}.png")
    # Resize by 1/4 for smaller file size (6 images is wide)
    montage = montage.resize((w * 6 // 4, h // 4))
    montage.save(out_path)
    print(f"Saved {out_path}")

shots_to_compare = [
    ("shot_006", "shot_005"),
    ("shot_010", "shot_009"),
    ("shot_036", "shot_035"),
    ("shot_004", "shot_003"),
]

for s, a in shots_to_compare:
    make_cross_weight_low_montage(s, a)

def make_rock_cycle_strip(weight_dir, label):
    rock_cycle_shots = ["shot_026", "shot_034", "shot_037", "shot_038", "shot_039", 
                        "shot_040", "shot_041", "shot_042", "shot_043", "shot_044"]
    images = []
    for s in rock_cycle_shots:
        img = Image.open(os.path.join(weight_dir, f"{s}.png"))
        images.append(img)
    
    w, h = images[0].size
    grid = Image.new("RGB", (w * 5, h * 2))
    for i, img in enumerate(images):
        row = i // 5
        col = i % 5
        grid.paste(img, (col * w, row * h))
    
    grid = grid.resize((w * 5 // 4, h * 2 // 4))
    grid_path = os.path.join(out_dir, f"strip_rock_cycle_{label}.png")
    grid.save(grid_path)
    print(f"Saved {grid_path}")

make_rock_cycle_strip(w02_dir, "w02")
