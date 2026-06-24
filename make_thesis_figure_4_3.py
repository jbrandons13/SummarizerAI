import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import matplotlib.patches as patches
import textwrap

# INPUTS
REFERENCE_IMAGE = "runs/geology/reference.png"

IMAGES = {
    "shot_005": {
        "own": "runs/geology/sweep/shot_005_w0.0.png",
        "fixed": "runs/geology/sweep/shot_005_w0.4.png",
        "adaptive": "runs/geology/sweep/shot_005_w0.2.png"
    },
    "shot_011": {
        "own": "runs/geology/sweep/shot_011_w0.0.png",
        "fixed": "runs/geology/sweep/shot_011_w0.4.png",
        "adaptive": "runs/geology/sweep/shot_011_w0.3.png"
    },
    "shot_013": {
        "own": "runs/geology/sweep/shot_013_w0.0.png",
        "fixed": "runs/geology/sweep/shot_013_w0.4.png",
        "adaptive": "runs/geology/sweep/shot_013_w0.5.png"
    }
}

PROMPTS = {
    "shot_005": "A wide establishing view of molten magma flowing and cooling to form distinctive igneous rock formations, flat 2D educational vector scene, vivid colors",
    "shot_011": "A wide landscape scene showing the effects of erosion, with weathered material being transported and shaping the contours of the land, flat 2D educational vector scene, vivid colors",
    "shot_013": "A wide view of the Earth, showing the global geological map and the various rock formations and processes that shape our planet, flat 2D educational vector scene, vivid colors"
}

CAPTIONS = {
    "shot_005": {
        "own": "w=0  |  content=1.00  |  concept=0.28",
        "fixed": "w=0.4  |  content=0.55  |  concept=0.33",
        "adaptive": "w=0.2  |  content=0.89  |  concept=0.30"
    },
    "shot_011": {
        "own": "w=0  |  content=1.00  |  concept=0.31",
        "fixed": "w=0.4  |  content=0.58  |  concept=0.36",
        "adaptive": "w=0.3  |  content=0.74  |  concept=0.34"
    },
    "shot_013": {
        "own": "w=0  |  content=1.00  |  concept=0.32",
        "fixed": "w=0.4  |  content=0.84  |  concept=0.35",
        "adaptive": "w=0.5  |  content=0.74  |  concept=0.37"
    }
}

SHOTS = ["shot_005", "shot_011", "shot_013"]
CONDITIONS = ["Prompt", "own", "fixed", "adaptive"]
COL_HEADERS = ["Prompt", "own (w0)", "fixed w0.4", "adaptive (per-shot)"]

def crop_center(img, target_aspect):
    h, w = img.shape[:2]
    aspect = w / h
    if aspect > target_aspect:
        # crop width
        new_w = int(h * target_aspect)
        start = (w - new_w) // 2
        return img[:, start:start+new_w]
    elif aspect < target_aspect:
        # crop height
        new_h = int(w / target_aspect)
        start = (h - new_h) // 2
        return img[start:start+new_h, :]
    return img

def truncate_prompt(text, max_words=12):
    words = text.split()
    if len(words) > max_words:
        return " ".join(words[:max_words]) + "..."
    return text

def main():
    # We use a 3x4 grid for the main shots. 
    # Increased height to 9.5 to leave a substantial top margin for the reference image.
    fig, axes = plt.subplots(3, 4, figsize=(13.0, 9.5), gridspec_kw={'width_ratios': [1.2, 2, 2, 2]})
    
    first_img = mpimg.imread(IMAGES["shot_005"]["own"])
    target_aspect = first_img.shape[1] / first_img.shape[0]

    for i, shot in enumerate(SHOTS):
        for j, cond in enumerate(CONDITIONS):
            ax = axes[i, j]
            
            ax.set_xticks([])
            ax.set_yticks([])
            for spine in ax.spines.values():
                spine.set_visible(False)
            
            if cond == "Prompt":
                prompt_text = truncate_prompt(PROMPTS[shot], max_words=14)
                wrapped_text = textwrap.fill(prompt_text, width=35)
                
                # Shot ID removed per user request
                # ax.text(0.0, 0.85, shot, transform=ax.transAxes, ha='left', va='bottom', fontsize=10, fontweight='bold')
                ax.text(0.0, 0.5, wrapped_text, transform=ax.transAxes, ha='left', va='center', fontsize=9.5)
                
                if i == 0:
                    ax.set_title(COL_HEADERS[j], fontsize=12, fontweight='bold', pad=15, loc='left')
                
            else:
                img_path = IMAGES[shot][cond]
                img = mpimg.imread(img_path)
                img_cropped = crop_center(img, target_aspect)
                ax.imshow(img_cropped)
                
                border_color = 'dodgerblue' if cond == 'adaptive' else 'gray'
                border_lw = 2.5 if cond == 'adaptive' else 0.5
                rect = patches.Rectangle((0, 0), img_cropped.shape[1], img_cropped.shape[0], 
                                         linewidth=border_lw, edgecolor=border_color, facecolor='none')
                ax.add_patch(rect)

                caption = CAPTIONS[shot][cond]
                ax.text(0.5, -0.06, caption, transform=ax.transAxes, ha='center', va='top', fontsize=9.5)
            
                if i == 0:
                    header = COL_HEADERS[j]
                    ax.set_title(header, fontsize=12, fontweight='bold', pad=15)

    key_line = "content = content preservation (DINOv2 similarity of the shot to its own no-anchoring image).  concept = CLIP-T concept alignment."
    fig.text(0.5, 0.02, key_line, ha='center', va='top', fontsize=9.5)

    # Adjust layout to leave a large top margin (top=0.68) for the reference image
    plt.subplots_adjust(left=0.02, right=0.98, top=0.68, bottom=0.08, wspace=0.05, hspace=0.25)
    
    # Force layout calculation so we can get bounding boxes
    fig.canvas.draw()
    
    # Get bounding box of the Prompt column's first cell (to align left edge)
    bbox_prompt = axes[0, 0].get_position()
    # Get bounding box of the first image cell (to match its exact width and height)
    bbox_img = axes[0, 1].get_position()
    
    # Create an inset axes in the top margin.
    # We place it in the top center of the entire figure.
    # Its size perfectly matches a grid cell image.
    ref_x = 0.5 - (bbox_img.width / 2)
    # Moved higher to 0.68 + 0.08 = 0.76 to avoid collision with column headers
    ref_ax = fig.add_axes([ref_x, 0.76, bbox_img.width, bbox_img.height])
    ref_ax.set_xticks([])
    ref_ax.set_yticks([])
    for spine in ref_ax.spines.values():
        spine.set_visible(False)
        
    ref_img = mpimg.imread(REFERENCE_IMAGE)
    ref_img_cropped = crop_center(ref_img, target_aspect)
    ref_ax.imshow(ref_img_cropped)
    # Add label below the reference image
    ref_ax.text(0.5, -0.05, "concept reference\n(all shots anchored here)", 
                transform=ref_ax.transAxes, ha='center', va='top', fontsize=9.5, fontweight='bold')

    # Figure title at the very top (REMOVED per user request)
    # fig.suptitle("Adaptive per-shot anchoring (blue) keeps each shot above its content floor", 
    #              fontsize=14, y=0.99)
    
    out_path = "V1_Geology_adaptive_grid.png"
    plt.savefig(out_path, dpi=300, facecolor='white', bbox_inches='tight')
    print(f"Saved to {out_path}")

if __name__ == "__main__":
    main()
