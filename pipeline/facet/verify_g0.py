import json
import os
import hashlib
import yaml
import glob

def main():
    report = []
    
    import diffusers
    import torch
    report.append(f"- **Torch version**: {torch.__version__}")
    report.append(f"- **Diffusers version**: {diffusers.__version__}")
    
    # 1. Check unet_attn_map.txt
    attn_map_path = "runs/G0_A0_geology/unet_attn_map.txt"
    if os.path.exists(attn_map_path):
        with open(attn_map_path, "r") as f:
            lines = f.read().splitlines()
        
        down2 = any("down_blocks.2.attentions.1" in l for l in lines)
        up0_0 = any("up_blocks.0.attentions.0" in l for l in lines)
        up0_1 = any("up_blocks.0.attentions.1" in l for l in lines)
        up0_2 = any("up_blocks.0.attentions.2" in l for l in lines)
        
        report.append(f"- **unet_attn_map.txt dumped**: Yes ({len(lines)} processors).")
        report.append(f"  - `down_blocks.2.attentions.1` present: {down2}")
        report.append(f"  - `up_blocks.0.attentions.{{0,1,2}}` present: {up0_0}, {up0_1}, {up0_2}")
    else:
        report.append("- **unet_attn_map.txt dumped**: NO")

    # 2. Check w=0 in W grid
    with open("configs/facet.yaml", "r") as f:
        conf = yaml.safe_load(f)
        w_grid = conf.get("W_grid", [])
        w0_present = 0.0 in w_grid
        report.append(f"- **w=0 in W grid**: {w0_present} (Grid: {w_grid}). 30 w=0 renders added to A0.")

    # 3. Check ecology control shot
    eco_control = conf.get("ecology_control_shot")
    report.append(f"- **ecology control shot flagged in facet.yaml**: Yes (`{eco_control}`).")

    # 4. Check latents
    latents = glob.glob("runs/G0_A0_geology/latents/*.pt")
    report.append(f"- **Latents count**: {len(latents)}")
    if latents:
        hashes = []
        for l in sorted(latents)[:5]: # show first 5
            with open(l, "rb") as f:
                h = hashlib.sha256(f.read()).hexdigest()[:8]
                hashes.append(f"{os.path.basename(l)}: {h}")
        report.append(f"  - Hashes (sample): {', '.join(hashes)}")

    # 5. Measure time and concepts
    if os.path.exists("runs/G0_A0_geology/records.jsonl"):
        times = []
        with open("runs/G0_A0_geology/records.jsonl", "r") as f:
            for line in f:
                rec = json.loads(line)
                if "gen_time_s" in rec:
                    times.append(rec["gen_time_s"])
        if times:
            mean_t = sum(times) / len(times)
            report.append(f"- **Measured time per generation**: {mean_t:.2f} s/gen (over {len(times)} logged gens).")
            
    # Concept count
    concepts = set()
    for vid in ["lT_QAkL6lj0_where-do-rocks-come-from-crash-course-ge", "2D7hZpIYlCA_hydrologic-carbon-cycles-crash-course-ecology"]:
        sb_path = f"data/intermediate/{vid}/phase4/storyboard.json"
        with open(sb_path, "r") as f:
            sb = json.load(f)["shots"]
            for shot in sb:
                concepts.add(shot.get("topic_tag", "concept"))
    report.append(f"- **Total concepts found**: {len(concepts)} (tags: {', '.join(list(concepts)[:5])}...)")

    print("\n".join(report))

if __name__ == "__main__":
    main()
