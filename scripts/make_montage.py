import os
from PIL import Image

base_dir = "data/intermediate/lT_QAkL6lj0_where-do-rocks-come-from-crash-course-ge/phase4"
never_chain_dir = os.path.join(base_dir, "never_chain", "images")
w08_dir = os.path.join(base_dir, "concept_anchor_canonical_w08", "images")
out_dir = os.path.join(base_dir, "_eval", "montages")
os.makedirs(out_dir, exist_ok=True)

def make_triplet(shot_id, anchor_id, label):
    img_baseline = Image.open(os.path.join(never_chain_dir, f"{shot_id}.png"))
    img_w08 = Image.open(os.path.join(w08_dir, f"{shot_id}.png"))
    img_anchor = Image.open(os.path.join(never_chain_dir, f"{anchor_id}.png"))
    
    # All images are 1344x768. Triplet is 3 * 1344 x 768.
    w, h = img_baseline.size
    montage = Image.new("RGB", (w * 3, h))
    montage.paste(img_baseline, (0, 0))
    montage.paste(img_w08, (w, 0))
    montage.paste(img_anchor, (w * 2, 0))
    
    out_path = os.path.join(out_dir, f"{label}_{shot_id}_to_{anchor_id}.png")
    # Resize by half for smaller file size
    montage = montage.resize((w * 3 // 2, h // 2))
    montage.save(out_path)
    print(f"Saved {out_path}")

# Suspect collapse
suspects = [("shot_036", "shot_035"), ("shot_044", "shot_026"), ("shot_040", "shot_026"), 
            ("shot_034", "shot_026"), ("shot_029", "shot_013"), ("shot_031", "shot_013")]
for s, a in suspects:
    make_triplet(s, a, "suspect")

# Valid cases
valids = [("shot_004", "shot_003"), ("shot_010", "shot_009"), ("shot_006", "shot_005")]
for s, a in valids:
    make_triplet(s, a, "valid")

# Strip concept
rock_cycle_shots = ["shot_026", "shot_034", "shot_037", "shot_038", "shot_039", 
                    "shot_040", "shot_041", "shot_042", "shot_043", "shot_044"]
images = []
for s in rock_cycle_shots:
    img = Image.open(os.path.join(w08_dir, f"{s}.png"))
    images.append(img)

w, h = images[0].size
# Let's make a grid 5 columns x 2 rows
grid = Image.new("RGB", (w * 5, h * 2))
for i, img in enumerate(images):
    row = i // 5
    col = i % 5
    grid.paste(img, (col * w, row * h))

# Resize by a quarter
grid = grid.resize((w * 5 // 4, h * 2 // 4))
grid_path = os.path.join(out_dir, "strip_rock_cycle.png")
grid.save(grid_path)
print(f"Saved {grid_path}")
