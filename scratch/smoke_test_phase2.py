import os
import sys
from pathlib import Path
import json
import logging

# Add project root to sys.path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from src.phase2_summarize import Phase2Summarizer
from src.models.llm_wrapper import GroqBackend
from src.utils.io import load_json_as_model
from src.schemas import SummaryScript

# Setup Logging
logging.basicConfig(level=logging.INFO)

def smoke_test():
    # Load .env
    env_path = project_root / ".env"
    if env_path.exists():
        with open(env_path, "r") as f:
            for line in f:
                if line.strip() and not line.startswith("#"):
                    key, _, value = line.partition("=")
                    os.environ[key.strip()] = value.strip()

    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        print("Error: GROQ_API_KEY not found")
        return

    backend = GroqBackend(api_key=api_key, model_name="llama-3.3-70b-versatile")
    summarizer = Phase2Summarizer(backend, {})

    transcript_path = project_root / "data/intermediate/tiny_video/transcript.json"
    if not transcript_path.exists():
        print(f"Error: Transcript not found at {transcript_path}")
        return

    print(f"Running Phase 2 smoke test for {transcript_path}...")
    output_path = summarizer.run(transcript_path, target_duration=90)
    print(f"Success! Output saved to {output_path}")

    with open(output_path, "r") as f:
        summary_data = json.load(f)
        print("\n--- PHASE 2 OUTPUT JSON ---")
        print(json.dumps(summary_data, indent=2))
        print("--- END OUTPUT ---\n")

if __name__ == "__main__":
    smoke_test()
