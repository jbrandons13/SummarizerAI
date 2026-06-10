from PIL import Image
import os

images = []
for shot in ["shot_005", "shot_011"]:
    for w in [0.0, 0.4]:
        for kind in ["orig", "runner"]:
            p = f"runs/probe/{shot}_w{w}_{kind}.png"
            if os.path.exists(p):
                images.append(Image.open(p))

if images:
    w, h = images[0].size
    montage = Image.new("RGB", (w * 4, h * 2))
    
    for i, img in enumerate(images):
        row = i // 4
        col = i % 4
        montage.paste(img, (col * w, row * h))
        
    montage.save("runs/probe/contact_sheet.png")
