from PIL import Image
import sys

def extract_strip(webp_path, out_path, num_frames_to_extract=6):
    img = Image.open(webp_path)
    
    frames = []
    try:
        while True:
            # We must convert to RGB because some webps are palette or RGBA
            frames.append(img.convert("RGB"))
            img.seek(len(frames))
    except EOFError:
        pass # End of sequence
        
    n_total = len(frames)
    print(f"Total frames found: {n_total}")
    
    if n_total == 0:
        print("No frames found!")
        return
        
    # Select evenly spaced frames
    indices = [int(i * (n_total - 1) / (num_frames_to_extract - 1)) for i in range(num_frames_to_extract)]
    print(f"Selected indices: {indices}")
    
    selected_frames = [frames[i] for i in indices]
    
    w, h = selected_frames[0].size
    strip = Image.new('RGB', (w * num_frames_to_extract, h))
    
    for i, frame in enumerate(selected_frames):
        strip.paste(frame, (i * w, 0))
        
    strip.save(out_path)
    print(f"Saved to {out_path}")

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: extract_webp_strip.py <in.webp> <out.png>")
        sys.exit(1)
    webp_in = sys.argv[1]
    strip_out = sys.argv[2]
    extract_strip(webp_in, strip_out)
