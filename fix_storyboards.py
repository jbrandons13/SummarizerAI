import json

def fix_sb(path):
    with open(path, "r") as f:
        data = json.load(f)
    for i, shot in enumerate(data["shots"]):
        shot["shot_id"] = f"shot_{i+1:03d}"
        shot["id"] = shot["shot_id"]
    with open(path, "w") as f:
        json.dump(data, f, indent=4)

fix_sb("data/intermediate/lT_QAkL6lj0_where-do-rocks-come-from-crash-course-ge/phase4/storyboard.json")
fix_sb("data/intermediate/2D7hZpIYlCA_hydrologic-carbon-cycles-crash-course-ecology/phase4/storyboard.json")
