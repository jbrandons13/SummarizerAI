import os
import hashlib
import json

def get_hash(path):
    if not os.path.exists(path): return "MISSING"
    if os.path.isdir(path): return "DIR"
    with open(path, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()[:8]

paths = [
    "pipeline/facet/runner.py",
    "pipeline/facet/scoring_wrap.py",
    "pipeline/facet/alt1.py",
    "pipeline/facet/centroid.py",
    "configs/facet.yaml",
    "pipeline/facet/seeds.json",
    "pipeline/facet/latents_cache",
    "pipeline/facet/ledger.json",
    "runs/G0_REPORT.md",
    "runs/probe/contact_sheet.png",
    "data/intermediate/lT_QAkL6lj0_where-do-rocks-come-from-crash-course-ge/phase4/storyboard.json",
    "data/intermediate/2D7hZpIYlCA_hydrologic-carbon-cycles-crash-course-ecology/phase4/storyboard.json"
]

inventory = {}
for p in paths:
    inventory[p] = {"exists": os.path.exists(p), "hash": get_hash(p)}

# Check 30 files in latents_cache
latents_count = 0
if os.path.isdir("pipeline/facet/latents_cache"):
    latents_count = len([f for f in os.listdir("pipeline/facet/latents_cache") if f.endswith(".pt")])
inventory["pipeline/facet/latents_cache"]["files"] = latents_count

with open("step0_inventory.json", "w") as f:
    json.dump(inventory, f, indent=2)

print("Inventory done")
