import os
import json
import subprocess
from PIL import Image
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

def make_qual():
    images_dir = "runs/geology/images_daca"
    shots = [f"shot_{i:03d}" for i in range(1, 15)]
    
    fig, axes = plt.subplots(2, 7, figsize=(20, 6))
    axes = axes.flatten()
    
    for i, shot in enumerate(shots):
        img_path = os.path.join(images_dir, f"{shot}.png")
        if os.path.exists(img_path):
            img = Image.open(img_path)
            axes[i].imshow(img)
        axes[i].axis('off')
        axes[i].set_title(shot, fontsize=10)
        
    plt.tight_layout()
    plt.savefig("WRITTING_STUFFS/fig4.1_qual.png", dpi=150)
    print("Saved fig4.1_qual.png")

def make_filmstrip():
    # We choose 4 shots for a filmstrip
    chosen_shots = ["shot_001", "shot_004", "shot_007", "shot_010"]
    
    with open("runs/geology/summary_script.json") as f:
        script = json.load(f)
    
    with open("runs/geology/storyboard.json") as f:
        storyboard = json.load(f)
        if "shots" in storyboard:
            storyboard = storyboard["shots"]
            
    with open("data/intermediate/lT_QAkL6lj0_where-do-rocks-come-from-crash-course-ge/phase4/shots.json") as f:
        shots_data = json.load(f)
        if "shots" in shots_data:
            shots_data = shots_data["shots"]
            
    fig, axes = plt.subplots(2, len(chosen_shots), figsize=(16, 7))
    raw_video = "data/raw_videos/lT_QAkL6lj0_where-do-rocks-come-from-crash-course-ge.mp4"
    
    for i, shot_id in enumerate(chosen_shots):
        # Find sentence id
        shot_info = next(s for s in shots_data if s["shot_id"] == shot_id)
        seg_id = int(shot_info["source_segment_ids"][0])
        sentence_info = next(s for s in script["sentences"] if s["id"] == seg_id)
        
        t_start, t_end = sentence_info["source_timestamp_hint"]
        t_mid = (t_start + t_end) / 2.0
        text = shot_info["text"]
        
        # Extract source frame
        out_frame = f"/tmp/{shot_id}_src.jpg"
        subprocess.run(["ffmpeg", "-y", "-ss", str(t_mid), "-i", raw_video, "-vframes", "1", "-q:v", "2", out_frame], 
                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        
        # Plot source frame
        if os.path.exists(out_frame):
            axes[0, i].imshow(Image.open(out_frame))
        axes[0, i].axis('off')
        if i == 0:
            axes[0, i].set_title("Source Video\n\n", fontsize=14, loc='left')
            
        # Plot generated frame
        gen_frame = f"runs/geology/images_daca/{shot_id}.png"
        if os.path.exists(gen_frame):
            axes[1, i].imshow(Image.open(gen_frame))
        axes[1, i].axis('off')
        if i == 0:
            axes[1, i].set_title("Generated Anchor\n\n", fontsize=14, loc='left')
            
        # Add caption
        import textwrap
        wrapped_text = "\n".join(textwrap.wrap(text, width=40))
        axes[1, i].text(0.5, -0.15, f"{shot_id}\n\"{wrapped_text}\"", transform=axes[1, i].transAxes, 
                        fontsize=10, ha='center', va='top', wrap=True)
                        
    plt.tight_layout()
    plt.savefig("WRITTING_STUFFS/fig4.2_filmstrip.png", dpi=150, bbox_inches='tight')
    print("Saved fig4.2_filmstrip.png")

if __name__ == "__main__":
    make_qual()
    make_filmstrip()
