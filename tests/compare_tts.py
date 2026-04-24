"""
Run: python tests/compare_tts.py
Output: 3 WAV files per sentence, side-by-side comparison
"""
import yaml
import os
from pathlib import Path
from src.phase3_voiceover import Phase3Voiceover
from src.utils.vram import VRAMManager
from src.utils.text import clean_for_tts

# Setup paths
output_dir = Path("data/output/tts_compare")
output_dir.mkdir(parents=True, exist_ok=True)

test_sentences = [
    "Most people believe eight hours of sleep is the magic number. New research says they are wrong!",
    "A Stanford study tracked ten thousand people over five years.",
    "This completely changes how we think about productivity.",
]

# Initialize VRAM Manager
vram_manager = VRAMManager(device_id=0)

# Test each backend
for backend_name in ["kokoro", "f5-tts", "chatterbox", "orpheus"]:
    print(f"\n>>> Testing Backend: {backend_name}")
    
    # Load config
    with open("configs/default.yaml", "r") as f:
        config = yaml.safe_load(f)
    
    # Override backend
    config["tts"]["backend"] = backend_name
    
    # Optional F5-TTS setup for comparison
    if backend_name == "f5-tts":
        # Use one of the kokoro outputs as reference if available
        ref_path = output_dir / "tts_compare_kokoro_0.wav"
        if ref_path.exists():
            config["tts"]["f5"] = {
                "ref_audio": str(ref_path),
                "ref_text": test_sentences[0]
            }
        else:
            print("  ! Skipping f5-tts: No reference audio (run kokoro first)")
            continue

    # Initialize Phase3Voiceover (it will auto-select the backend)
    try:
        p3 = Phase3Voiceover(config, vram_manager)
        backend = p3.backend
        
        for i, text in enumerate(test_sentences):
            cleaned_text = clean_for_tts(text)
            filename = f"tts_compare_{backend_name}_{i}.wav"
            output_path = output_dir / filename
            
            print(f"Generating ({i+1}/{len(test_sentences)}): {cleaned_text[:50]}...")
            duration = backend.generate(cleaned_text, output_path)
            
            if output_path.exists():
                print(f"  ✓ Saved: {output_path} ({duration:.2f}s)")
            else:
                print(f"  ✗ Failed to save: {output_path}")
        
        # Explicitly unload
        backend.unload()
        print(f"Backend {backend_name} unloaded.")
        
    except Exception as e:
        print(f"Error testing backend {backend_name}: {e}")

print("\nDone. Listen to files in data/output/tts_compare/*.wav")
