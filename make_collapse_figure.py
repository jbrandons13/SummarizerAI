import argparse
import json
import os
import csv
from PIL import Image, ImageDraw, ImageFont
import math

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--compute-scores", action="store_true")
    parser.add_argument("--reference", required=False)
    args = parser.parse_args()

    with open(args.manifest, "r") as f:
        manifest = json.load(f)

    # Group by shot
    shots = {}
    for entry in manifest:
        shot_id = entry["shot_id"]
        if shot_id not in shots:
            shots[shot_id] = []
        shots[shot_id].append(entry)

    # Sort weights
    for shot_id in shots:
        shots[shot_id].sort(key=lambda x: x["weight"])

    scores_map = {}
    if args.compute_scores:
        # Load the precomputed CSV instead of running DINOv2
        csv_path = os.path.join(os.path.dirname(args.manifest), "collapse_metrics.csv")
        if os.path.exists(csv_path):
            with open(csv_path, "r") as f:
                reader = csv.reader(f)
                for row in reader:
                    if len(row) >= 3 and row[0].startswith("shot_"):
                        # row[0] is shot, row[1] is weight, row[2] is sim_to_ref
                        shot = row[0].strip('"')
                        weight = float(row[1])
                        sim = float(row[2])
                        scores_map[(shot, weight)] = sim

    # Set up drawing
    cell_w, cell_h = 416, 240
    padding = 20
    header_h = 60
    
    unique_weights = sorted(list(set([entry["weight"] for entry in manifest])))
    cols = 1 + len(unique_weights) # Ref + weights
    rows = len(shots)
    
    img_w = cols * cell_w + (cols + 1) * padding
    img_h = header_h + rows * cell_h + (rows + 1) * padding
    
    grid = Image.new("RGB", (img_w, img_h), "white")
    draw = ImageDraw.Draw(grid)
    
    try:
        font = ImageFont.truetype("DejaVuSans-Bold.ttf", 20)
    except:
        font = ImageFont.load_default()
        
    # Draw headers
    draw.text((padding + cell_w//2, padding), "Reference", fill="black", font=font, anchor="mt")
    for j, w in enumerate(unique_weights):
        x = padding + (j+1)*(cell_w+padding) + cell_w//2
        draw.text((x, padding), f"Weight: {w:.1f}", fill="black", font=font, anchor="mt")
        
    if args.reference:
        ref_img = Image.open(args.reference).convert("RGB")
    
    # Draw rows
    for i, (shot_id, entries) in enumerate(shots.items()):
        y = header_h + padding + i*(cell_h+padding)
        
        # Draw Reference (Col 0)
        x = padding
        if args.reference:
            r_img = ref_img.resize((cell_w, cell_h), Image.LANCZOS)
            grid.paste(r_img, (x, y))
            draw.text((x + 10, y + 10), "Reference", fill="white", font=font)
            
        # Draw Weights (Col 1..N)
        for j, entry in enumerate(entries):
            x = padding + (j+1)*(cell_w+padding)
            try:
                c_img = Image.open(entry["path"]).convert("RGB")
                c_img = c_img.resize((cell_w, cell_h), Image.LANCZOS)
                
                # Compute score if needed
                score = entry.get("dino", 0.0)
                if args.compute_scores:
                    w_key = float(entry["weight"])
                    if (shot_id, w_key) in scores_map:
                        score = scores_map[(shot_id, w_key)]
                    
                grid.paste(c_img, (x, y))
                
                # Outline high weights
                if entry["weight"] >= 0.6:
                    draw.rectangle([x, y, x+cell_w, y+cell_h], outline="red", width=5)
                    
                # Text label
                text = f"{shot_id}\nW={entry['weight']:.1f}"
                if score > 0:
                    text += f"\nDINOv2: {score:.3f}"
                    
                draw.text((x + 10, y + 10), text, fill="white", font=font)
            except Exception as e:
                print(f"Error loading {entry['path']}: {e}")

    grid.save(args.out)
    print(f"Saved figure to {args.out}")

if __name__ == "__main__":
    main()
