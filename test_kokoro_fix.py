import sys
import os
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).parent))

from src.models.tts_wrapper import KokoroBackend

def test_tts():
    # Setup paths from your config
    model_path = "models/kokoro/kokoro-v1.0.onnx"
    voices_path = "models/kokoro/voices-v1.0.bin"
    
    if not os.path.exists(model_path):
        print(f"Error: Model not found at {model_path}")
        return

    print(f"Loading Kokoro TTS (Model: {model_path})...")
    backend = KokoroBackend(model_path=model_path, voices_path=voices_path)
    
    text = "The speaker discusses how AI changes the job market."
    output_path = "test_bella.wav"
    
    print(f"Generating audio for: '{text}' using voice: af_bella")
    backend.generate(text, output_path, voice="af_bella")
    
    if os.path.exists(output_path):
        print(f"Success! Audio saved to {output_path}")
    else:
        print("Failed to generate audio.")

if __name__ == "__main__":
    test_tts()
