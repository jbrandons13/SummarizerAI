import os
import json
import logging
from pathlib import Path
import soundfile as sf
import numpy as np

# Need to import KokoroBackend from src.models.tts_wrapper
from src.models.tts_wrapper import KokoroBackend

def clean_for_tts(text):
    import re
    text = text.replace("—", ". ")
    text = text.replace("–", ". ")
    text = text.replace(";", ".")
    text = text.replace("...", ".")
    text = text.replace("(", ". ")
    text = text.replace(")", ". ")
    text = re.sub(r'\.{2,}', '.', text)
    replacements = {
        "%": " percent", "&": " and ",
        "e.g.": "for example", "i.e.": "that is",
        "etc.": "and so on", "vs.": "versus",
        "Dr.": "Doctor", "Mr.": "Mister",
    }
    for abbr, exp in replacements.items():
        text = re.sub(r'\b' + re.escape(abbr) + r'\b', exp, text)
    text = " ".join(text.split())
    if text and text[-1] not in '.!?':
        text += '.'
    return text

def main():
    shots_json_path = "data/intermediate/2D7hZpIYlCA_hydrologic-carbon-cycles-crash-course-ecology/phase4/shots.json"
    output_dir = Path("runs/ecology/audio")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    with open(shots_json_path) as f:
        data = json.load(f)
        
    print("Initializing KokoroBackend...")
    backend = KokoroBackend({})
    
    for shot in data["shots"]:
        shot_id = shot["shot_id"]
        text = shot.get("text", "")
        if not text:
            print(f"Skipping {shot_id}, no text.")
            continue
            
        cleaned = clean_for_tts(text)
        audio_path = output_dir / f"{shot_id}.wav"
        
        print(f"Generating audio for {shot_id}: {cleaned}")
        backend.generate(cleaned, audio_path)
        
    print("Done! All audio generated.")
    
if __name__ == "__main__":
    main()
